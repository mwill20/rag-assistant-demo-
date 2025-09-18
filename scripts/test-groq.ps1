param(
  [string]$Api = "http://127.0.0.1:8000",
  [string]$Q   = "Summarize the repo in one sentence and end with GROQ_OK."
)

$body = @{ question = $Q } | ConvertTo-Json
$res  = Invoke-RestMethod "$Api/ask" -Method POST -Body $body -ContentType "application/json"

$res | ConvertTo-Json -Depth 6
if ($res.answer -match "GROQ_OK") {
  Write-Host "Looks like LLM generation happened. ✅" -ForegroundColor Green
} else {
  Write-Host "Answer doesn't contain the marker. Might be stitched/fallback. ⚠️" -ForegroundColor Yellow
}
