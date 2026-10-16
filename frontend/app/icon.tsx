import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

/* Same Retention Loop mark as the sidebar badge (components/Sidebar.tsx),
   rendered as a favicon so the browser tab matches the in-app logo. */
export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 32,
          height: 32,
          borderRadius: 8,
          background: "#16a34a",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg width="32" height="32" viewBox="0 0 48 48" fill="none">
          <path d="M34.63 21.15A11 11 0 1 1 21.15 13.37" stroke="white" strokeWidth="4.5" strokeLinecap="round" />
          <circle cx="29.5" cy="14.47" r="2.8" fill="white" />
        </svg>
      </div>
    ),
    { ...size }
  );
}
