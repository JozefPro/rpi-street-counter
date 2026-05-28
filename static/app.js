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
  totalCounted: document.getElementById("total-counted"),
  latestCrossingEvent: document.getElementById("latest-crossing-event"),
  inferenceMs: document.getElementById("inference-ms"),
  detectionFps: document.getElementById("detection-fps"),
  detectionFrameId: document.getElementById("detection-frame-id"),
  boxesDrawnCount: document.getElementById("boxes-drawn-count"),
  annotatedStream: document.getElementById("annotated-stream"),
  countingStatus: document.getElementById("counting-status"),
  activeTracks: document.getElementById("active-tracks"),
  streamDelay: document.getElementById("stream-delay"),
  latestFrameId: document.getElementById("latest-frame-id"),
  streamedFrameId: document.getElementById("streamed-frame-id"),
  frameBufferSize: document.getElementById("frame-buffer-size"),
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
    fields.totalCounted.textContent = status.total_counted ?? 0;
    fields.latestCrossingEvent.textContent = status.latest_crossing_event || "none";
    fields.inferenceMs.textContent =
      status.inference_ms === null || status.inference_ms === undefined
        ? "n/a"
        : `${formatNumber(status.inference_ms)} ms`;
    fields.detectionFps.textContent = formatNumber(status.detection_fps);
    fields.detectionFrameId.textContent = status.detection_frame_id ?? "n/a";
    fields.boxesDrawnCount.textContent = status.boxes_drawn_count ?? 0;
    fields.annotatedStream.textContent = status.stream_uses_annotated_frame ? "yes" : "no";
    fields.countingStatus.textContent = status.counting_enabled ? "on" : "off";
    fields.activeTracks.textContent = status.active_tracks ?? 0;
    fields.streamDelay.textContent = `${status.stream_delay_frames ?? 0} frames`;
    fields.latestFrameId.textContent = status.latest_frame_id ?? "n/a";
    fields.streamedFrameId.textContent = status.streamed_frame_id ?? "n/a";
    fields.frameBufferSize.textContent = status.frame_buffer_size ?? "n/a";
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
