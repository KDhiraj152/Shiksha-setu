#!/bin/bash
# Update documentation footers with author info
# Usage: ./scripts/utils/update-doc-footers.sh

AUTHOR_BLOCK='
---

## 👨‍💻 Author

**K Dhiraj**

- 📧 Email: [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com)
- 🐙 GitHub: [@KDhiraj152](https://github.com/KDhiraj152)
- 💼 LinkedIn: [K Dhiraj](https://www.linkedin.com/in/k-dhiraj-83b025279/)

---

*Last updated: November 2025*
'

# Find all project markdown files (exclude node_modules, venv, .git)
find_md_files() {
    find /Users/kdhiraj_152/Downloads/shiksha_setu -name "*.md" -type f \
        ! -path "*/node_modules/*" \
        ! -path "*/.git/*" \
        ! -path "*/venv/*" \
        ! -path "*/.pytest_cache/*" \
        ! -name "README.md" \
        2>/dev/null
}

echo "Documentation footer update script"
echo "==================================="
echo "Author: K Dhiraj"
echo "Email: k.dhiraj.srihari@gmail.com"
echo "GitHub: @KDhiraj152"
echo ""
