const fields = {
  activeModel: document.getElementById("active-model"),
  detectionStatus: document.getElementById("detection-status"),
  fps: document.getElementById("fps"),
  carsLeft: document.getElementById("cars-left"),
  carsRight: document.getElementById("cars-right"),
  cpu: document.getElementById("cpu"),
  ram: document.getElementById("ram"),
  temperature: document.getElementById("temperature"),
  cameraError: document.getElementById("camera-error"),
};

function formatNumber(value, fallback = "0") {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(1) : fallback;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const status = await response.json();
    fields.activeModel.textContent = status.active_model || "none";
    fields.detectionStatus.textContent = status.detection_enabled ? "on" : "off";
    fields.fps.textContent = formatNumber(status.fps);
    fields.carsLeft.textContent = status.cars_left ?? 0;
    fields.carsRight.textContent = status.cars_right ?? 0;
    fields.cpu.textContent = `${formatNumber(status.cpu_percent)}%`;
    fields.ram.textContent = `${formatNumber(status.ram_percent)}%`;
    fields.temperature.textContent =
      status.temperature_c === null ? "n/a" : `${formatNumber(status.temperature_c)} C`;

    if (status.camera_error) {
      fields.cameraError.hidden = false;
      fields.cameraError.textContent = status.camera_error;
    } else {
      fields.cameraError.hidden = true;
      fields.cameraError.textContent = "";
    }
  } catch (error) {
    fields.cameraError.hidden = false;
    fields.cameraError.textContent = `Status unavailable: ${error.message}`;
  }
}

refreshStatus();
setInterval(refreshStatus, 1000);
