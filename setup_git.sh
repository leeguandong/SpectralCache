#!/bin/bash
# Quick setup script for SpectralCache GitHub repository

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          SpectralCache GitHub Setup                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Initialize git repository
echo "📦 Initializing Git repository..."
git init

# Add all files
echo "📝 Adding files..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: SpectralCache

- Complete paper (main.tex, appendix.tex)
- 6 benchmark scripts for reproducing all experiments
- Example usage script
- Comprehensive documentation (README, IMPLEMENTATION, CONTRIBUTING)
- All paper figures and qualitative comparisons
- Apache 2.0 license"

echo ""
echo "✅ Git repository initialized!"
echo ""
echo "🎯 Next steps:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Create a new repository on GitHub:"
echo "   https://github.com/new"
echo ""
echo "2. Add remote and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/SpectralCache.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Update README.md with:"
echo "   - Your GitHub username in clone URL"
echo "   - arXiv link when paper is published"
echo "   - Your contact email"
echo ""
echo "4. (Optional) Add GitHub badges:"
echo "   - Stars, forks, license badges"
echo "   - CI/CD status badges"
echo ""
echo "✨ Your project is ready for GitHub!"
