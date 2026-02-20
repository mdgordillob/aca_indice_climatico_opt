import subprocess
import sys
import os

tex_file = sys.argv[1] if len(sys.argv) > 1 else 'AnnualICAForecast_Report.tex'
tex_path = os.path.abspath(tex_file)

print(f"LaTeX file: {tex_path}")
print("")
print("=" * 80)
print("PDFLATEX NOT INSTALLED ON THIS SYSTEM")
print("=" * 80)
print("")
print("To compile the PDF, choose one of these options:")
print("")
print("1. ONLINE (Recommended - no installation needed):")
print("   - Upload AnnualICAForecast_Report.tex to Overleaf.com")
print("   - Click 'Recompile' to generate PDF")
print("   - Download the PDF")
print("")
print("2. INSTALL LOCALLY:")
print("   Option A (Windows): Install MikTeX from https://miktex.org/")
print("   Option B (Windows): Install TeX Live from https://www.tug.org/texlive/")
print("   Option C (Chocolatey): choco install miktex-portable")
print("")
print("3. COMMAND LINE (after installation):")
print(f"   pdflatex -interaction=nonstopmode {tex_file}")
print("   bibtex AnnualICAForecast_Report")
print("   pdflatex -interaction=nonstopmode {tex_file}")
print("")
print("=" * 80)
print("")

# Show file location
print(f"File location: {tex_path}")
print(f"File size: {os.path.getsize(tex_path) / 1024:.1f} KB")
