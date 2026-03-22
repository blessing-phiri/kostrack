// Kostrack shared nav + logo
// Include with: <script src="/assets/nav.js"></script>
// Then call: renderNav(activePage) where activePage = 'docs'|'github'|''

const LOGO_ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none" style="width:28px;height:28px;flex-shrink:0;">
  <defs>
    <linearGradient id="nibg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#0D1B2E"/><stop offset="100%" stop-color="#152540"/></linearGradient>
    <linearGradient id="nibar" x1="0" y1="1" x2="0" y2="0"><stop offset="0%" stop-color="#1E3A5F"/><stop offset="100%" stop-color="#2E5480"/></linearGradient>
    <linearGradient id="niline" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#F5A623"/><stop offset="100%" stop-color="#E8891A"/></linearGradient>
  </defs>
  <rect width="64" height="64" rx="14" fill="url(#nibg)"/>
  <rect x="8"  y="42" width="9" height="12" rx="2" fill="url(#nibar)" opacity="0.45"/>
  <rect x="20" y="34" width="9" height="20" rx="2" fill="url(#nibar)" opacity="0.65"/>
  <rect x="32" y="24" width="9" height="30" rx="2" fill="url(#nibar)" opacity="0.85"/>
  <rect x="44" y="14" width="9" height="40" rx="2" fill="url(#nibar)"/>
  <polyline points="12.5,46 24.5,38 36.5,27 48.5,17" stroke="url(#niline)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
  <circle cx="48.5" cy="17" r="4" fill="#F5A623"/>
  <line x1="6" y1="56" x2="58" y2="56" stroke="#F5A623" stroke-width="1.5" stroke-linecap="round" opacity="0.25"/>
</svg>`;

function renderNav(activePage = '') {
  const root = document.querySelector('nav.nav') || (() => {
    const n = document.createElement('nav');
    n.className = 'nav';
    document.body.prepend(n);
    return n;
  })();

  root.innerHTML = `
    <a href="/" class="nav-logo">
      ${LOGO_ICON_SVG}
      <span class="nav-wordmark">kos<span>track</span></span>
    </a>
    <ul class="nav-links">
      <li><a href="/docs/" class="${activePage === 'docs' ? 'active' : ''}">Docs</a></li>
      <li><a href="https://github.com/bphiri/kostrack" target="_blank">GitHub</a></li>
      <li><a href="https://pypi.org/project/kostrack" target="_blank" class="nav-cta">PyPI →</a></li>
    </ul>
  `;
}

function renderSidebar(activePage = '') {
  const nav = [
    { label: 'Getting Started', links: [
      { href: '/docs/',                     title: 'Introduction',      id: 'intro' },
      { href: '/docs/quickstart.html',      title: 'Quick Start',       id: 'quickstart' },
      { href: '/docs/docker.html',          title: 'Docker Setup',      id: 'docker' },
      { href: '/docs/environment.html',     title: 'Environment Vars',  id: 'environment' },
    ]},
    { label: 'SDK Reference', links: [
      { href: '/docs/configure.html',       title: 'configure()',        id: 'configure' },
      { href: '/docs/anthropic.html',       title: 'Anthropic',          id: 'anthropic' },
      { href: '/docs/openai.html',          title: 'OpenAI',             id: 'openai' },
      { href: '/docs/gemini.html',          title: 'Gemini',             id: 'gemini' },
      { href: '/docs/deepseek.html',        title: 'DeepSeek',           id: 'deepseek' },
      { href: '/docs/tracing.html',         title: 'Tracing & Spans',    id: 'tracing' },
      { href: '/docs/tags.html',            title: 'Attribution Tags',   id: 'tags' },
    ]},
    { label: 'Governance', links: [
      { href: '/docs/budgets.html',         title: 'Budget Enforcement', id: 'budgets' },
      { href: '/docs/pricing-sync.html',    title: 'Pricing Sync',       id: 'pricing-sync' },
    ]},
    { label: 'Platform', links: [
      { href: '/docs/platform-api.html',    title: 'Platform API',       id: 'platform-api' },
      { href: '/docs/cli.html',             title: 'CLI Reference',      id: 'cli' },
    ]},
    { label: 'Dashboard & Queries', links: [
      { href: '/docs/grafana.html',         title: 'Grafana Overview',   id: 'grafana' },
      { href: '/docs/queries.html',         title: 'Useful Queries',     id: 'queries' },
    ]},
    { label: 'Integrations', links: [
      { href: '/docs/fastapi.html',         title: 'FastAPI',            id: 'fastapi' },
      { href: '/docs/langgraph.html',       title: 'LangGraph',          id: 'langgraph' },
    ]},
  ];

  const sidebar = document.querySelector('.sidebar');
  if (!sidebar) return;

  sidebar.innerHTML = nav.map(section => `
    <div class="sidebar-section">
      <span class="sidebar-label">${section.label}</span>
      ${section.links.map(l => `
        <a href="${l.href}" class="sidebar-link ${activePage === l.id ? 'active' : ''}">${l.title}</a>
      `).join('')}
    </div>
  `).join('');
}
