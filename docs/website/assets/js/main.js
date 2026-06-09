// Nexum v2.0 Documentation Website - Main JavaScript

// Mobile menu toggle
function toggleMobileMenu() {
    const menu = document.getElementById('mobile-menu');
    menu.classList.toggle('hidden');
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
            
            // Close mobile menu if open
            const mobileMenu = document.getElementById('mobile-menu');
            if (!mobileMenu.classList.contains('hidden')) {
                mobileMenu.classList.add('hidden');
            }
        }
    });
});

// Navbar background change on scroll
window.addEventListener('scroll', function() {
    const nav = document.querySelector('nav');
    if (window.scrollY > 50) {
        nav.classList.add('bg-nexum-primary/90');
        nav.classList.remove('glass');
    } else {
        nav.classList.remove('bg-nexum-primary/90');
        nav.classList.add('glass');
    }
});

// Active navigation link highlighting
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (scrollY >= sectionTop - 200) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('text-nexum-accentLight');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('text-nexum-accentLight');
        }
    });
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate-fade-in');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe all feature cards and sections
document.querySelectorAll('.feature-card, section > div').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Add fade-in animation styles
const style = document.createElement('style');
style.textContent = `
    .animate-fade-in {
        opacity: 1 !important;
        transform: translateY(0) !important;
    }
`;
document.head.appendChild(style);

// Copy code functionality
document.querySelectorAll('.code-block').forEach(block => {
    const copyBtn = document.createElement('button');
    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
    copyBtn.className = 'absolute top-2 right-2 bg-nexum-accent hover:bg-nexum-accentDark text-white px-3 py-1 rounded text-xs transition-colors';
    copyBtn.style.position = 'absolute';
    block.style.position = 'relative';
    block.appendChild(copyBtn);
    
    copyBtn.addEventListener('click', () => {
        const code = block.querySelector('code').textContent;
        navigator.clipboard.writeText(code).then(() => {
            copyBtn.innerHTML = '<i class="fas fa-check"></i>';
            setTimeout(() => {
                copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
            }, 2000);
        });
    });
});

// Performance metrics animation
function animateValue(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const value = Math.floor(progress * (end - start) + start);
        element.textContent = value;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Animate performance metrics when visible
const performanceObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const metrics = entry.target.querySelectorAll('.text-4xl');
            metrics.forEach(metric => {
                const text = metric.textContent;
                const number = parseInt(text);
                if (!isNaN(number)) {
                    animateValue(metric, 0, number, 1500);
                }
            });
            performanceObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

const performanceSection = document.getElementById('performance');
if (performanceSection) {
    performanceObserver.observe(performanceSection);
}

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const mobileMenu = document.getElementById('mobile-menu');
        if (!mobileMenu.classList.contains('hidden')) {
            mobileMenu.classList.add('hidden');
        }
    }
});

// Loading animation
window.addEventListener('load', () => {
    document.body.classList.add('loaded');
});

// Add loaded class styles
const loadedStyle = document.createElement('style');
loadedStyle.textContent = `
    body {
        opacity: 0;
        transition: opacity 0.5s ease;
    }
    body.loaded {
        opacity: 1;
    }
`;
document.head.appendChild(loadedStyle);

// Console welcome message
console.log('%c Nexum v2.0 ', 'background: linear-gradient(90deg, #3b82f6, #2563eb); color: white; padding: 10px 20px; border-radius: 5px; font-size: 16px; font-weight: bold;');
console.log('%c Bridging Human and Machine Intelligence ', 'color: #60a5fa; font-size: 12px;');
console.log('%c Documentation: https://docs.nexum.ai ', 'color: #94a3b8; font-size: 10px;');
