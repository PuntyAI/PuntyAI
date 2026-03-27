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
          dark: '#131318',
          card: '#1b1b20',
          card2: '#35343a',
          border: 'rgba(255,255,255,0.06)',
          magenta: '#ff46a0',
          mbright: '#ffb0cc',
          cyan: '#3cd7ff',
          orange: '#ffb59d',
          yellow: '#ffc107',
          purple: '#9d4edd',
          muted: '#808090',
          coral: '#ff46a0',
          gold: '#ffc107',
          pink: '#ffb0cc',
          'pink-dark': '#b30069',
        }
      },
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        heading: ['"Space Grotesk"', 'sans-serif'],
        body: ['"Inter"', 'sans-serif'],
      },
      borderRadius: {
        'sm': '0.125rem',
        DEFAULT: '0.25rem',
        'md': '0.375rem',
        'lg': '0.375rem',
      }
    }
  },
  plugins: [],
}
