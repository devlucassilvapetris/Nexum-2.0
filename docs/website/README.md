# Nexum v2.0 Documentation Website

Modern, responsive documentation website for Nexum v2.0 - Hybrid Operating System.

## Features

- **Modern Design**: Clean, professional interface with gradient backgrounds and glass morphism effects
- **Responsive Layout**: Fully responsive design that works on all devices
- **Smooth Animations**: Fade-in effects, floating elements, and smooth scrolling
- **Interactive Elements**: Mobile menu, code copy functionality, and animated metrics
- **Comprehensive Documentation**: Complete coverage of all AI/ML features
- **Performance Metrics**: Visual representation of system performance benchmarks

## Structure

```
docs/website/
├── index.html          # Main HTML file
├── assets/
│   ├── css/           # Custom CSS (if needed)
│   └── js/
│       └── main.js    # JavaScript for interactivity
└── README.md          # This file
```

## Usage

### Local Development

Simply open `index.html` in a web browser:

```bash
# Using Python
cd docs/website
python -m http.server 8000

# Using Node.js
cd docs/website
npx serve

# Or just open the file directly in your browser
```

### Deployment

The website is static and can be deployed to any static hosting service:

- **GitHub Pages**: Push to gh-pages branch
- **Netlify**: Connect your repository
- **Vercel**: Import your project
- **AWS S3**: Upload to S3 bucket with static website hosting

## Customization

### Colors

The color scheme is defined in the Tailwind config within `index.html`:

```javascript
colors: {
    nexum: {
        primary: '#0f172a',
        secondary: '#1e293b',
        accent: '#3b82f6',
        accentLight: '#60a5fa',
        accentDark: '#2563eb',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
    }
}
```

### Fonts

The website uses:
- **Inter**: Main font for body text
- **JetBrains Mono**: Monospace font for code

Both are loaded from Google Fonts.

### Animations

Custom animations are defined in the Tailwind config:
- `gradient`: Background gradient animation
- `float`: Floating animation for hero elements
- `pulse-slow`: Slow pulse animation

## Sections

1. **Hero**: Introduction with animated gradient background
2. **Overview**: System architecture layers
3. **AI/ML Stack**: Comprehensive AI/ML features documentation
4. **Performance**: Benchmarks and efficiency metrics
5. **Installation**: Setup instructions
6. **API Reference**: Code examples for all major APIs
7. **Footer**: Links and resources

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Dependencies

- TailwindCSS (via CDN)
- Font Awesome (via CDN)
- Google Fonts (via CDN)

No build process required - everything works out of the box.

## License

MIT License - See main project LICENSE file for details.
