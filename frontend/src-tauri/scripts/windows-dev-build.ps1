<#
.SYNOPSIS
    Wrapper for `cargo check`/`cargo build` on Windows that sets up the environment
    quirks discovered while first getting this crate to compile/link on Windows (see
    docs/adr/0018-vendoring-whisper-rs-sys-bindgen-072-bug-msvc.md and
    docs/adr/0019-crt-statico-whisper-cpp-windows-conflitto-sherpa-onnx-sys.md in the
    docs workspace, outside this repo).

.DESCRIPTION
    Two of the three problems found on that first Windows build are fixed permanently in
    code (vendored whisper-rs-sys: bindgen bumped to 0.72, static CRT forced on MSVC via
    build.rs) and need nothing from this script. What's left is genuinely per-machine and
    can't be committed as a fixed path/value:

      1. bindgen needs libclang from a Visual Studio install with the actual C++ toolset
         (not a standalone LLVM.org install, and not a VS install with only the Clang
         *tools* component but no MSVC compiler) -- this script searches known VS install
         locations for it instead of hardcoding one machine's path.
      2. cmake's own Visual-Studio-instance detection can fail to resolve a valid
         generator on some multi-VS-install machines (observed with a very new VS
         version alongside an older one) -- if `ninja` is on PATH, this script forces
         CMAKE_GENERATOR=Ninja to sidestep that detection entirely. Not forced globally
         via committed Cargo config, because that would newly require Ninja on
         macOS/Linux too where it was never needed.

    Also warns (does not act automatically) if the repo appears to be under a
    OneDrive-synced folder, since that caused spurious build-cache lock errors on the
    machine this was first debugged on -- see the ADRs above for the recommended fix
    (CARGO_TARGET_DIR outside the synced tree).

.PARAMETER CargoCommand
    "check" (default) or "build". Anything else is passed through to `cargo` as-is
    (e.g. "build --release").

.EXAMPLE
    .\windows-dev-build.ps1
    .\windows-dev-build.ps1 build
#>

param(
    [string]$CargoCommand = "check"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$CrateDir = Join-Path $RepoRoot "frontend\src-tauri"

Write-Host "== Meetily Windows dev build helper ==" -ForegroundColor Cyan

# --- 1. Find a Visual Studio install with the real MSVC C++ toolset (not just Clang tools) ---
$vsRoots = @(
    "C:\Program Files\Microsoft Visual Studio",
    "C:\Program Files (x86)\Microsoft Visual Studio"
)
$msvcToolsetDirs = $vsRoots | Where-Object { Test-Path $_ } | ForEach-Object {
    Get-ChildItem -Path $_ -Recurse -Depth 4 -Filter "cl.exe" -ErrorAction SilentlyContinue
} | Where-Object { $_.FullName -match "\\Hostx64\\x64\\cl\.exe$" }

if (-not $msvcToolsetDirs) {
    Write-Warning "No cl.exe found under any Visual Studio install. Install the 'Desktop development with C++' workload (see docs/adr/0019 in the docs workspace) before continuing."
} else {
    Write-Host "Found MSVC toolset(s):" -ForegroundColor Green
    $msvcToolsetDirs | ForEach-Object { Write-Host "  $($_.FullName)" }
}

# --- 2. Find a matching libclang.dll (bindgen needs this, not just cl.exe) ---
$libclangCandidates = $vsRoots | Where-Object { Test-Path $_ } | ForEach-Object {
    Get-ChildItem -Path $_ -Recurse -Depth 6 -Filter "libclang.dll" -ErrorAction SilentlyContinue
} | Where-Object { $_.FullName -match "\\x64\\bin\\libclang\.dll$" }

if ($libclangCandidates) {
    $libclangDir = ($libclangCandidates | Select-Object -First 1).DirectoryName
    $env:LIBCLANG_PATH = $libclangDir
    Write-Host "LIBCLANG_PATH = $libclangDir" -ForegroundColor Green
} else {
    Write-Warning "No libclang.dll found under a Visual Studio install (VC\Tools\Llvm\x64\bin). Add the 'C++ Clang Compiler for Windows' individual component in Visual Studio Installer, or bindgen will fall back to a possibly-incompatible libclang found elsewhere on PATH."
}

# --- 3. Prefer Ninja for CMake if available, to avoid cmake's own VS-instance-detection bugs ---
$ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue
if ($ninja) {
    $env:CMAKE_GENERATOR = "Ninja"
    Write-Host "CMAKE_GENERATOR = Ninja ($($ninja.Source))" -ForegroundColor Green
} else {
    Write-Warning "ninja.exe not found on PATH -- letting CMake auto-detect a generator. If the build fails with 'could not find any instance of Visual Studio' for a generator like 'Visual Studio NN 20XX', install Ninja (winget install Ninja-build.Ninja) and re-run this script."
}

# --- 4. Warn about OneDrive-synced repo paths (observed to cause stale/locked build-cache dirs) ---
if ($RepoRoot -match "OneDrive") {
    Write-Warning "Repo path is under a OneDrive-synced folder ($RepoRoot). This caused spurious build-cache locks (cargo clean failing mid-cleanup, stale linked artifacts) on the machine this was first debugged on. Consider setting CARGO_TARGET_DIR to a path outside OneDrive, e.g.:`n  `$env:CARGO_TARGET_DIR = 'C:\meetily-build-target'"
}

# --- 5. Run cargo ---
Write-Host "`nRunning: cargo $CargoCommand (in $CrateDir)" -ForegroundColor Cyan
Push-Location $CrateDir
try {
    Invoke-Expression "cargo $CargoCommand"
} finally {
    Pop-Location
}
