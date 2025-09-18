param(
  [string]$Pattern = "test_prompt_security.py"
)
Write-Host "Running security tests..." -ForegroundColor Cyan
pytest "tests\$Pattern" -q
