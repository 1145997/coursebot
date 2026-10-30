$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$version = "1.9.3"
$stageName = "CourseBot-v$version-win64"
$releaseRoot = Join-Path $PSScriptRoot "release"
$stage = Join-Path $releaseRoot $stageName
$zip = Join-Path $releaseRoot "$stageName.zip"

Write-Host "Cleaning old output..."
if (Test-Path "build") { Remove-Item "build" -Recurse -Force }
if (Test-Path "dist") { Remove-Item "dist" -Recurse -Force }
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

Write-Host "Building CourseBot.exe..."
cmd /c "python -m PyInstaller --noconfirm --clean CourseBot.spec > build.log 2>&1"
if ($LASTEXITCODE -ne 0) {
    Get-Content build.log -Tail 40
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path "dist\CourseBot\CourseBot.exe")) {
    throw "CourseBot.exe not found"
}

Write-Host "Staging release..."
Copy-Item "dist\CourseBot\*" $stage -Recurse -Force
Copy-Item "README.md" $stage -Force
Copy-Item "RELEASE_NOTES.md" $stage -Force
Copy-Item "packaging\README_RELEASE.txt" $stage -Force

foreach ($name in @("config.json", "logs")) {
    $path = Join-Path $stage $name
    if (Test-Path $path) { Remove-Item $path -Recurse -Force }
}

Write-Host "Creating zip..."
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -CompressionLevel Optimal

$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
"$hash  $stageName.zip" | Set-Content (Join-Path $releaseRoot "SHA256SUMS.txt") -Encoding ASCII

Write-Host "Done:"
Write-Host "  Folder: $stage"
Write-Host "  Zip:    $zip"
Write-Host "  SHA256: $hash"



