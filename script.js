const views = {
    dashboard: "Command Center",
    upload: "Upload Media",
    analysis: "AI Analysis",
    results: "Results Dashboard",
    explain: "Explainable AI",
    reports: "Reports",
    history: "Analysis History",
    settings: "Settings"
};

const activities = [
    ["DG-2847", "interview_clip.mp4", "Likely fake", "97.2%"],
    ["DG-2846", "press_photo.png", "Verified real", "98.6%"],
    ["DG-2845", "voiceover_reel.mov", "Needs review", "78.4%"],
    ["DG-2844", "profile_image.jpg", "Likely fake", "93.1%"]
];

const pipeline = [
    "File Validation",
    "Face Detection",
    "Frame Extraction",
    "Feature Analysis",
    "Deepfake Detection",
    "Explainability",
    "Report Creation"
];

const signals = [
    ["Facial texture anomaly", 97],
    ["Temporal landmark drift", 91],
    ["Mouth-shape mismatch", 87],
    ["Metadata provenance risk", 74]
];

// =====================
// Navigation
// =====================

function switchView(id) {
    document.querySelectorAll(".view").forEach(view =>
        view.classList.toggle("active", view.id === id)
    );

    document.querySelectorAll(".nav-item").forEach(item =>
        item.classList.toggle("active", item.dataset.view === id)
    );

    const title = document.querySelector("#page-title");

    if (title) {
        title.textContent = views[id];
    }
}

document.querySelectorAll("[data-view], [data-view-jump]").forEach(control => {
    control.addEventListener("click", () =>
        switchView(control.dataset.view || control.dataset.viewJump)
    );
});

// =====================
// Dashboard Widgets
// =====================

const bars = document.querySelector(".bars");

if (bars) {
    bars.innerHTML = [44, 58, 51, 72, 65, 88, 76, 91, 64, 70, 83, 92, 87, 96]
        .map(value =>
            `<span class="bar" style="height:${value}%"></span>`
        )
        .join("");
}

const activityList = document.querySelector(".activity-list");

if (activityList) {
    activityList.innerHTML = activities
        .map(([id, file, verdict, confidence]) =>
            `<div class="activity">
                <b>${id}</b>
                <span>${file}</span>
                <span>${verdict}</span>
                <strong>${confidence}</strong>
            </div>`
        )
        .join("");
}

const pipelineContainer = document.querySelector(".pipeline");

if (pipelineContainer) {
    pipelineContainer.innerHTML = pipeline
        .map((name, index) =>
            `<div class="stage ${index < 4 ? "done" : index === 4 ? "active" : ""}">
                <b>${name}</b>
                <div class="progress">
                    <span style="width:${index < 4 ? 100 : index === 4 ? 62 : 0}%"></span>
                </div>
            </div>`
        )
        .join("");
}

const signalList = document.querySelector(".signal-list");

if (signalList) {
    signalList.innerHTML = signals
        .map(([name, value]) =>
            `<div class="signal">
                <b>${name}</b>
                <div class="signal-meter">
                    <span style="width:${value}%"></span>
                </div>
                <strong>${value}%</strong>
            </div>`
        )
        .join("");
}

// =====================
// History Table
// =====================

function renderHistory(filter = "") {

    const rows = [
        ["DG-2847", "interview_clip.mp4", "Likely fake", "97.2%", "Escalated"],
        ["DG-2846", "press_photo.png", "Verified real", "98.6%", "Closed"],
        ["DG-2845", "voiceover_reel.mov", "Needs review", "78.4%", "Review"],
        ["DG-2844", "profile_image.jpg", "Likely fake", "93.1%", "Reported"],
        ["DG-2843", "board_meeting.mov", "Verified real", "95.0%", "Closed"],
        ["DG-2842", "evidence_frame.png", "Likely fake", "89.7%", "Escalated"]
    ];

    const normalized = filter.toLowerCase();

    const historyRows = document.querySelector("#historyRows");

    if (!historyRows) return;

    historyRows.innerHTML = rows
        .filter(row =>
            row.join(" ").toLowerCase().includes(normalized)
        )
        .map(row =>
            `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`
        )
        .join("");
}

renderHistory();

const historyFilter = document.querySelector("#historyFilter");

if (historyFilter) {
    historyFilter.addEventListener("input", e =>
        renderHistory(e.target.value)
    );
}

// =====================
// DeepGuard AI Upload
// =====================

const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#fileInput");
const fileStatus = document.querySelector("#fileStatus");
const selectFileBtn = document.querySelector("#selectFile");

if (selectFileBtn) {
    selectFileBtn.addEventListener("click", () => {
        fileInput.click();
    });
}

// =====================
// AI Prediction Function
// =====================

async function analyzeImage(file) {

    if (!file) return;

    try {

        fileStatus.textContent =
            `Uploading ${file.name}...`;

        const formData = new FormData();
        formData.append("image", file);

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",
                body: formData
            }
        );

        if (!response.ok) {
            throw new Error("Prediction failed");
        }

        const result = await response.json();

        fileStatus.textContent =
            `Analysis completed successfully`;

        // Update UI
        const predictionElement =
            document.getElementById("prediction");

        const confidenceElement =
            document.getElementById("confidence");

        if (predictionElement) {
            predictionElement.textContent =
                result.prediction;
        }

        if (confidenceElement) {
            confidenceElement.textContent =
                result.confidence + "%";
        }

        console.log("Prediction:", result);

        alert(
            `Prediction: ${result.prediction}\nConfidence: ${result.confidence}%`
        );

    } catch (error) {

        console.error(error);

        fileStatus.textContent =
            "Connection to DeepGuard AI failed";

        alert(
            "Could not connect to Flask backend.\nMake sure app.py is running."
        );
    }
}

// =====================
// File Upload
// =====================

if (fileInput) {

    fileInput.addEventListener("change", async () => {

        const file = fileInput.files[0];

        if (!file) {
            fileStatus.textContent = "No file selected";
            return;
        }

        fileStatus.textContent =
            `${file.name} ready for analysis`;

        await analyzeImage(file);
    });
}

// =====================
// Drag & Drop
// =====================

if (dropzone) {

    ["dragenter", "dragover"].forEach(type => {
        dropzone.addEventListener(type, event => {
            event.preventDefault();
            dropzone.classList.add("drag");
        });
    });

    ["dragleave", "drop"].forEach(type => {
        dropzone.addEventListener(type, event => {
            event.preventDefault();
            dropzone.classList.remove("drag");
        });
    });

    dropzone.addEventListener("drop", async event => {

        const file = event.dataTransfer.files[0];

        if (!file) return;

        fileStatus.textContent =
            `${file.name} dropped successfully`;

        await analyzeImage(file);
    });
}