/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: "#080b13",
          panel: "#10141f",
          soft: "#0d1220",
          sidebar: "#0a0d17",
        },
        border: {
          subtle: "#1e2436",
        },
        accent: {
          purple: "#7c6ff0",
          purpleLight: "#9b8fff",
          blue: "#4f8cff",
        },
        good: {
          DEFAULT: "#22c55e",
          soft: "#173321",
        },
        bad: {
          DEFAULT: "#f2555c",
          soft: "#3a1a1e",
        },
        warn: {
          DEFAULT: "#f5a524",
        },
        muted: "#8891a5",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.02) inset, 0 8px 24px -12px rgba(0,0,0,0.5)",
      },
    },
  },
  plugins: [],
};
