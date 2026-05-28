const fields = {
  activeModel: document.getElementById("active-model"),
  activeModelCard: document.getElementById("active-model-card"),
  detectionStatus: document.getElementById("detection-status"),
  detectionCardStatus: document.getElementById("detection-card-status"),
  fpsSummary: document.getElementById("fps-summary"),
  fps: document.getElementById("fps"),
  requestedResolution: document.getElementById("requested-resolution"),
  actualResolution: document.getElementById("actual-resolution"),
  carsLeft: document.getElementById("cars-left"),
  carsRight: document.getElementById("cars-right"),
  visibleVehicles: document.getElementById("visible-vehicles"),
  inferenceMs: document.getElementById("inference-ms"),
  detectionFps: document.getElementById("detection-fps"),
  detectionErrorCard: document.getElementById("detection-error-card"),
  detectionError: document.getElementById("detection-error"),
  cpu: document.getElementById("cpu"),
  ram: document.getElementById("ram"),
  temperature: document.getElementById("temperature"),
  cameraError: document.getElementById("camera-error"),
};

function formatNumber(value, fallback = "0") {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(1) : fallback;
}

function formatResolution(width, height) {
  const parsedWidth = Number(width);
  const parsedHeight = Number(height);

  if (!Number.isFinite(parsedWidth) || !Number.isFinite(parsedHeight)) {
    return "n/a";
  }

  if (parsedWidth <= 0 || parsedHeight <= 0) {
    return "n/a";
  }

  return `${Math.round(parsedWidth)}x${Math.round(parsedHeight)}`;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const status = await response.json();
    fields.activeModel.textContent = status.active_model || "none";
    fields.activeModelCard.textContent = status.active_model || "none";
    const detectionText = status.detection_enabled ? "on" : "off";
    fields.detectionStatus.textContent = detectionText;
    fields.detectionCardStatus.textContent = detectionText;
    const measuredFps = formatNumber(status.measured_stream_fps ?? status.fps);
    fields.fpsSummary.textContent = measuredFps;
    fields.fps.textContent = measuredFps;
    fields.requestedResolution.textContent = formatResolution(
      status.requested_width,
      status.requested_height,
    );
    fields.actualResolution.textContent = formatResolution(
      status.actual_width,
      status.actual_height,
    );
    fields.carsLeft.textContent = status.cars_left ?? 0;
    fields.carsRight.textContent = status.cars_right ?? 0;
    fields.visibleVehicles.textContent = status.vehicle_detections_count ?? 0;
    fields.inferenceMs.textContent =
      status.inference_ms === null || status.inference_ms === undefined
        ? "n/a"
        : `${formatNumber(status.inference_ms)} ms`;
    fields.detectionFps.textContent = formatNumber(status.detection_fps);
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

    if (status.detection_error) {
      fields.detectionErrorCard.hidden = false;
      fields.detectionError.textContent = status.detection_error;
    } else {
      fields.detectionErrorCard.hidden = true;
      fields.detectionError.textContent = "";
    }
  } catch (error) {
    fields.cameraError.hidden = false;
    fields.cameraError.textContent = `Status unavailable: ${error.message}`;
  }
}

refreshStatus();
setInterval(refreshStatus, 1000);
