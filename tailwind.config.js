/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './punty/web/templates/**/*.html',
    './punty/public/templates/**/*.html',
  ],
  theme: {
    extend: {
      colors: {
        punty: {
          dark: '#0a0a0f',
          card: '#12121a',
          card2: '#1a1a25',
          border: 'rgba(255,255,255,0.08)',
          magenta: '#e91e8c',
          mbright: '#ff2d9e',
          cyan: '#00d4ff',
          orange: '#ff6b35',
          yellow: '#ffc107',
          purple: '#9d4edd',
          muted: '#606070',
          coral: '#e91e8c',
          gold: '#ffc107',
        }
      },
      fontFamily: {
        display: ['"Orbitron"', 'sans-serif'],
        heading: ['"Rajdhani"', 'sans-serif'],
        body: ['"Source Sans Pro"', 'sans-serif'],
      }
    }
  },
  plugins: [],
}
