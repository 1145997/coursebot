param([string]$Version = "", [string]$Label = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $Version) {
    $versionSource = Get-Content -LiteralPath "coursebot\__init__.py" -Raw
    $Version = [regex]::Match($versionSource, '__version__ = "([^"]+)"').Groups[1].Value
}
if ($Version -notmatch '^[A-Za-z0-9.-]+$' -or $Label -notmatch '^[A-Za-z0-9.-]*$') { throw "Invalid version or label" }
$suffix = if ($Label) { "-$Label" } else { "" }
$stageName = "CourseBot-v$Version$suffix-win64"
$releaseRoot = Join-Path $PSScriptRoot "release"
$stage = Join-Path $releaseRoot $stageName
$zip = Join-Path $releaseRoot "$stageName.zip"

if ((Test-Path -LiteralPath $stage) -or (Test-Path -LiteralPath $zip)) {
    throw "Release already exists; use -Label to preserve the baseline."
}
$buildRoot = Join-Path $PSScriptRoot "build\$stageName"
$distRoot = Join-Path $PSScriptRoot "dist\$stageName"
New-Item -ItemType Directory -Path $releaseRoot -Force | Out-Null

Write-Host "Building CourseBot.exe..."
$buildLog = Join-Path $releaseRoot "$stageName-build.log"
$ErrorActionPreference = "Continue" # PyInstaller 的正常 INFO 也写入 stderr。
& python -m PyInstaller --noconfirm --clean --distpath $distRoot --workpath $buildRoot CourseBot.spec *> $buildLog
$buildExitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($buildExitCode -ne 0) {
    Get-Content -LiteralPath $buildLog -Tail 40
    throw "PyInstaller failed with exit code $buildExitCode"
}
$builtApp = Join-Path $distRoot "CourseBot"
if (-not (Test-Path -LiteralPath (Join-Path $builtApp "CourseBot.exe"))) {
    throw "CourseBot.exe not found"
}

Write-Host "Staging release..."
New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -Path (Join-Path $builtApp "*") -Destination $stage -Recurse
Copy-Item "README.md" $stage -Force
Copy-Item "RELEASE_NOTES.md" $stage -Force
Copy-Item "packaging\README_RELEASE.txt" $stage -Force

New-Item -ItemType Directory -Path (Join-Path $stage "docs") | Out-Null
Copy-Item -LiteralPath "docs\VALIDATION.md" -Destination (Join-Path $stage "docs")
$commit = (& git rev-parse HEAD).Trim()
$dirty = [bool](& git status --porcelain --untracked-files=no)
@{ version=$Version; commit=$commit; local_changes=$dirty; remote_published=$false;
   settings_included=$false; created=(Get-Date).ToString("s") } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stage "BUILD_INFO.json") -Encoding UTF8

Write-Host "Creating zip..."
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -CompressionLevel Optimal

$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
"$hash  $stageName.zip" | Set-Content -LiteralPath (Join-Path $releaseRoot "$stageName.sha256") -Encoding ASCII

Write-Host "Done:"
Write-Host "  Folder: $stage"
Write-Host "  Zip:    $zip"
Write-Host "  SHA256: $hash"



