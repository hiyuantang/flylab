import { useEffect, useRef } from "react";
import type { SensoryFrame } from "../lib/types";

// An angular receptor map, not a reconstruction of subjective visual experience.
// Canvas avoids thousands of React/SVG elements on each live update.
export function CompoundEye({
  eye,
  side,
  channel,
}: {
  eye: SensoryFrame["vision"]["eyes"][number];
  side: number;
  channel: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = canvas.current?.getContext("2d");
    if (!ctx) return;
    const width = 720,
      height = 540;
    // Front faces inward between the panels; only the display is mirrored.
    const displayX = (azimuth: number) => {
      const x = (azimuth + 30) * 3;
      return side === 0 ? width - x : x;
    };
    ctx.fillStyle = "#101b23";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#30404c";
    ctx.lineWidth = 1;
    for (let az = 0; az <= 180; az += 90) {
      ctx.beginPath();
      ctx.moveTo(displayX(az), 0);
      ctx.lineTo(displayX(az), height);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(0, 270);
    ctx.lineTo(width, 270);
    ctx.stroke();
    const values = eye.channels[channel];
    eye.angles_degrees.forEach(([azimuth, elevation], i) => {
      let az = azimuth * (side === 0 ? 1 : -1);
      if (az < -30) az += 360;
      const value = Math.round(255 * Math.pow(Math.max(0, values[i]), 1 / 2.2));
      ctx.fillStyle = `rgb(${value},${value},${value})`;
      ctx.beginPath();
      ctx.ellipse(
        displayX(az),
        (90 - elevation) * 3,
        5.2,
        5.2,
        0,
        0,
        Math.PI * 2,
      );
      ctx.fill();
    });
  }, [eye, side, channel]);
  return (
    <figure className="compound-eye">
      <figcaption>
        {side === 0 ? "Left" : "Right"} compound eye{" "}
        <span>{eye.count} measured viewing directions</span>
      </figcaption>
      <canvas
        ref={canvas}
        width={720}
        height={540}
        role="img"
        aria-label={`${side === 0 ? "Left" : "Right"} compound eye ${channel} angular receptor map`}
      />
      <div className="retina-axis">
        <span>{side === 0 ? "180° rear" : "0° front"}</span>
        <span>90° side</span>
        <span>{side === 0 ? "0° front" : "180° rear"}</span>
      </div>
      <small>Dorsal ↑ · ventral ↓ · brightness shown in grayscale</small>
    </figure>
  );
}
