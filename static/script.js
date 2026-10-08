/* =========================================================
   AI Crop Decision Assistant - Frontend logic
   Plain JavaScript (no framework) - kept simple for viva.
   ========================================================= */

let lastPrediction = null; // stores the last /predict response, used by What-If tab

// ---------------- Tab switching ----------------
function switchTab(tabId) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(tabId).classList.add("active");
  document.querySelector(`.tab-btn[data-tab="${tabId}"]`).classList.add("active");
}

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

// ---------------- Formatting helpers ----------------
function formatRupees(value) {
  return "₹" + Number(value).toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

// ---------------- Dashboard ----------------
async function loadDashboard() {
  try {
    const res = await fetch("/dashboard-data");
    const data = await res.json();
    if (data.error) return;

    document.getElementById("statCrops").textContent = data.total_crops;
    document.getElementById("statRecords").textContent = data.total_records;
    document.getElementById("statYield").textContent = data.avg_yield;
    document.getElementById("statProfit").textContent = formatRupees(data.avg_profit);

    drawBarChart("yieldChart", data.crops, data.yield_by_crop, "#2F6B3C", "Quintals/acre");
    drawBarChart("profitChart", data.crops, data.profit_by_crop, "#C4901F", "₹/acre");
    drawBarChart("demandChart", data.crops, data.demand_by_crop, "#7A5230", "Demand score (0-100)");
  } catch (err) {
    console.error("Failed to load dashboard data:", err);
  }
}

const chartInstances = {};
function drawBarChart(canvasId, labels, values, color, axisLabel) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
  chartInstances[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{ label: axisLabel, data: values, backgroundColor: color, borderRadius: 6 }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } },
    },
  });
}

// ---------------- Recommend form ----------------
const form = document.getElementById("predictForm");
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorBox = document.getElementById("formError");
  const btn = document.getElementById("predictBtn");
  errorBox.textContent = "";

  const payload = {
    state: form.state.value,
    district: form.district.value,
    soil_type: form.soil_type.value,
    season: form.season.value,
    land_size: form.land_size.value,
    budget: form.budget.value,
    water_availability: form.water_availability.value,
    previous_crop: form.previous_crop.value,
  };

  btn.disabled = true;
  btn.textContent = "Analyzing...";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    let data;
    try {
      data = await res.json();
    } catch (parseErr) {
      console.error("Server response was not valid JSON:", parseErr);
      errorBox.textContent = `Server returned an unexpected response (status ${res.status}). Check the terminal running 'python app.py' for the error.`;
      return;
    }

    if (!res.ok) {
      errorBox.textContent = data.error || `Something went wrong (status ${res.status}). Please try again.`;
      return;
    }

    lastPrediction = data;
    renderResults(data);
    prepareWhatIf(data);
  } catch (err) {
    console.error("Fetch to /predict failed:", err);
    errorBox.textContent = "Could not reach the server. Make sure 'python app.py' is running and you opened http://127.0.0.1:5000 in this browser.";
  } finally {
    btn.disabled = false;
    btn.textContent = "Get Recommendation";
  }
});

function riskCardClass(risk) {
  return "metric-card risk-" + risk;
}

function renderResults(data) {
  const r = data.recommended;
  const alternatives = data.alternatives;
  const area = document.getElementById("resultsArea");

  let html = `
    <div class="hero-result">
      <div>
        <div class="crop-name-label">Recommended Crop</div>
        <div class="crop-name">${r.crop}</div>
      </div>
      <div class="suitability-score">
        <div class="num">${r.suitability}%</div>
        <div class="lbl">Crop Suitability Score</div>
      </div>
    </div>

    <div class="metric-grid">
      <div class="metric-card"><div class="m-label">Expected Yield</div><div class="m-value">${r.expected_yield} Q/acre</div></div>
      <div class="metric-card"><div class="m-label">Estimated Revenue</div><div class="m-value">${formatRupees(r.estimated_revenue)}</div></div>
      <div class="metric-card"><div class="m-label">Estimated Cost</div><div class="m-value">${formatRupees(r.estimated_cost)}</div></div>
      <div class="metric-card"><div class="m-label">Estimated Profit</div><div class="m-value">${formatRupees(r.estimated_profit)}</div></div>
      <div class="${riskCardClass(r.risk_level)}"><div class="m-label">Risk Level</div><div class="m-value">${r.risk_level}</div></div>
      <div class="metric-card"><div class="m-label">Water Requirement</div><div class="m-value">${r.water_requirement}</div></div>
      <div class="metric-card"><div class="m-label">Market Demand</div><div class="m-value">${r.market_demand}/100</div></div>
      <div class="metric-card"><div class="m-label">Market Price</div><div class="m-value">${formatRupees(r.market_price_per_quintal)}/Q</div></div>
    </div>

    <div class="note-box">
      <strong>AI-based estimated fertilizer recommendation:</strong>
      <ul>${r.fertilizer_tips.map(t => `<li>${t}</li>`).join("")}</ul>
      <div style="margin-top:8px;">Actual fertilizer application should be based on local agricultural expert / soil-test guidance.</div>
    </div>

    ${!r.affordable_within_budget ? `<div class="note-box" style="border-left-color:#B3452C;">Note: Estimated cost for this crop is higher than your entered budget. Consider a smaller land area or check the alternative crops below.</div>` : ""}

    <div class="section-block">
      <h3>Alternative Crops Comparison</h3>
      <table class="compare-table">
        <thead>
          <tr><th>Crop</th><th>Suitability</th><th>Yield</th><th>Profit</th><th>Risk</th><th>Demand</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>${r.crop} (Recommended)</strong></td>
            <td>${r.suitability}%</td>
            <td>${r.expected_yield} Q/acre</td>
            <td>${formatRupees(r.estimated_profit)}</td>
            <td>${r.risk_level}</td>
            <td>${r.market_demand}/100</td>
          </tr>
          ${alternatives.map(a => `
          <tr>
            <td>${a.crop}</td>
            <td>${a.suitability}%</td>
            <td>${a.expected_yield} Q/acre</td>
            <td>${formatRupees(a.estimated_profit)}</td>
            <td>${a.risk_level}</td>
            <td>${a.market_demand}/100</td>
          </tr>`).join("")}
        </tbody>
      </table>
      <div class="chart-card compare-chart">
        <h3>Profit Comparison (₹/acre)</h3>
        <canvas id="compareProfitChart"></canvas>
      </div>
    </div>

    <div class="disclaimer-footer">
      Predictions are estimates based on historical/sample data and should be validated with local
      agricultural experts and current market conditions.
    </div>
  `;

  area.innerHTML = html;

  const allCrops = [r, ...alternatives];
  drawBarChart(
    "compareProfitChart",
    allCrops.map(c => c.crop),
    allCrops.map(c => c.estimated_profit),
    "#2F6B3C",
    "₹/acre"
  );

  area.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------- What-If tab ----------------
function prepareWhatIf(data) {
  document.getElementById("whatifEmpty").style.display = "none";
  document.getElementById("whatifContent").style.display = "block";
}

const slider = document.getElementById("irrigationSlider");
slider.addEventListener("input", () => {
  document.getElementById("irrigationValue").textContent = slider.value;
});

document.getElementById("runWhatIfBtn").addEventListener("click", async () => {
  if (!lastPrediction) {
    alert("Please get a crop recommendation first on the Recommend tab.");
    return;
  }

  const btn = document.getElementById("runWhatIfBtn");
  btn.disabled = true;
  btn.textContent = "Calculating...";

  const payload = {
    crop: lastPrediction.recommended.crop,
    raw_row: lastPrediction.raw_row,
    land_size: lastPrediction.input_summary.land_size,
    budget: lastPrediction.input_summary.budget,
    irrigation_increase_pct: Number(slider.value),
  };

  try {
    const res = await fetch("/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    let data;
    try {
      data = await res.json();
    } catch (parseErr) {
      console.error("Server response was not valid JSON:", parseErr);
      alert(`Server returned an unexpected response (status ${res.status}). Check the terminal running 'python app.py'.`);
      return;
    }

    if (!res.ok) {
      alert(data.error || `Could not run what-if analysis (status ${res.status}).`);
      return;
    }
    renderWhatIf(data);
  } catch (err) {
    console.error("Fetch to /what-if failed:", err);
    alert("Could not reach the server. Make sure 'python app.py' is running.");
  } finally {
    btn.disabled = false;
    btn.textContent = "Recalculate";
  }
});

function renderWhatIfBox(elId, r) {
  document.getElementById(elId).innerHTML = `
    <div class="wf-row"><span>Yield</span><span>${r.expected_yield} Q/acre</span></div>
    <div class="wf-row"><span>Suitability</span><span>${r.suitability}%</span></div>
    <div class="wf-row"><span>Estimated Profit</span><span>${formatRupees(r.estimated_profit)}</span></div>
    <div class="wf-row"><span>Risk Level</span><span>${r.risk_level}</span></div>
  `;
}

function renderWhatIf(data) {
  renderWhatIfBox("whatifBefore", data.before);
  renderWhatIfBox("whatifAfter", data.after);

  const diff = data.difference;
  const yieldCls = diff.yield_change >= 0 ? "diff-up" : "diff-down";
  const profitCls = diff.profit_change >= 0 ? "diff-up" : "diff-down";
  const suitCls = diff.suitability_change >= 0 ? "diff-up" : "diff-down";

  document.getElementById("whatifDiff").innerHTML = `
    <div class="diff-line">Yield change: <span class="${yieldCls}">${diff.yield_change >= 0 ? "+" : ""}${diff.yield_change} Q/acre</span></div>
    <div class="diff-line">Profit change: <span class="${profitCls}">${diff.profit_change >= 0 ? "+" : ""}${formatRupees(diff.profit_change)}</span></div>
    <div class="diff-line">Suitability change: <span class="${suitCls}">${diff.suitability_change >= 0 ? "+" : ""}${diff.suitability_change}%</span></div>
  `;
}

// ---------------- Init ----------------
loadDashboard();
