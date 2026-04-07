const historyBody = document.getElementById('historyBody');
const currentStatus = document.getElementById('currentStatus');
const currentKeyword = document.getElementById('currentKeyword');
const statusBox = document.getElementById('statusBox');

async function fetchHistory() {
    try {
        const response = await fetch(`/history?ts=${Date.now()}`);

        const json = await response.json();

        historyBody.innerHTML = "";

        if (json.data.length === 0) {
            currentStatus.innerText = "LISTENING";
            currentKeyword.innerText = "Waiting for STOP...";
            return;
        }

        const latest = json.data[0];

        if (latest.status === "DANGER") {
            statusBox.className = "current-status-box danger";
            currentStatus.innerText = "🚨 STOP DETECTED";
        } else {
            statusBox.className = "current-status-box normal";
            currentStatus.innerText = "LISTENING";
        }

        currentKeyword.innerText =
            `Detected: ${latest.keyword_detected} (${(latest.confidence * 100).toFixed(1)}%)`;

        json.data.forEach(item => {
            const tr = document.createElement("tr");

            tr.innerHTML = `
                <td>${item.timestamp}</td>
                <td><span class="status-badge ${item.status.toLowerCase()}">${item.status}</span></td>
                <td><strong>${item.keyword_detected}</strong></td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
            `;

            historyBody.appendChild(tr);
        });

    } catch (err) {
        console.error("Fetch error:", err);
    }
}

// 🔁 FAST REFRESH
setInterval(fetchHistory, 800);
fetchHistory();