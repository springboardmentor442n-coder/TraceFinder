// frontend/tailwind.config.cjs
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'Helvetica', 'Arial'],
      },
      colors: {
        primary: {
          50: '#eef2ff',
          100: '#e0e7ff',
          300: '#c7d2fe',
          500: '#6366f1',
          700: '#4f46e5'
        },
        pastel: {
          blue: '#eef8ff',
          pink: '#fff0f6'
        }
      },
      boxShadow: {
        soft: '0 6px 18px rgba(15,23,42,0.06)',
      }
    },
  },
  plugins: [],
};
