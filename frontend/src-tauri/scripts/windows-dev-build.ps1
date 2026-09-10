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
    # Real path is e.g. <root>\<version-or-year-folder>\<Edition>\VC\Tools\MSVC\<toolset>\bin\Hostx64\x64\cl.exe
    # -- 9 directory levels below $_. -Depth 4 (the original value here) never reached it on
    # any real install; bumped with margin so a differently-named version folder (e.g. "18")
    # doesn't reopen the same bug.
    Get-ChildItem -Path $_ -Recurse -Depth 10 -Filter "cl.exe" -ErrorAction SilentlyContinue
} | Where-Object { $_.FullName -match "\\Hostx64\\x64\\cl\.exe$" }

if (-not $msvcToolsetDirs) {
    Write-Warning "No cl.exe found under any Visual Studio install. Install the 'Desktop development with C++' workload (see docs/adr/0019 in the docs workspace) before continuing."
} else {
    Write-Host "Found MSVC toolset(s):" -ForegroundColor Green
    $msvcToolsetDirs | ForEach-Object { Write-Host "  $($_.FullName)" }
}

# --- 2. Find a matching libclang.dll (bindgen needs this, not just cl.exe) ---
$libclangCandidates = $vsRoots | Where-Object { Test-Path $_ } | ForEach-Object {
    # <root>\<version-or-year-folder>\<Edition>\VC\Tools\Llvm\x64\bin\libclang.dll -- 7 levels
    # below $_. -Depth 6 (the original value here) was one level short; same fix as cl.exe above.
    Get-ChildItem -Path $_ -Recurse -Depth 10 -Filter "libclang.dll" -ErrorAction SilentlyContinue
} | Where-Object { $_.FullName -match "\\x64\\bin\\libclang\.dll$" }

if ($libclangCandidates) {
    $libclangDir = ($libclangCandidates | Select-Object -First 1).DirectoryName
    $env:LIBCLANG_PATH = $libclangDir
    Write-Host "LIBCLANG_PATH = $libclangDir" -ForegroundColor Green
} else {
    Write-Warning "No libclang.dll found under a Visual Studio install (VC\Tools\Llvm\x64\bin). Add the 'C++ Clang Compiler for Windows' individual component in Visual Studio Installer, or bindgen will fall back to a possibly-incompatible libclang found elsewhere on PATH."
}

# --- 3. Import INCLUDE/LIB from vcvarsall.bat (bindgen needs these to find stdbool.h and
#        the other C standard headers -- LIBCLANG_PATH alone only tells it which clang to
#        load, not where to look for headers; found the hard way: "libclang.dll" loads fine
#        but bindgen still fails with "'stdbool.h' file not found" without this). ---
if ($msvcToolsetDirs) {
    $vsInstallRoot = ($msvcToolsetDirs | Select-Object -First 1).FullName -replace '\\VC\\Tools\\MSVC\\.*$', ''
    $vcvarsall = Join-Path $vsInstallRoot "VC\Auxiliary\Build\vcvarsall.bat"
    if (Test-Path $vcvarsall) {
        Write-Host "Importing INCLUDE/LIB from $vcvarsall (x64)..." -ForegroundColor Cyan
        $envLines = cmd /c "`"$vcvarsall`" x64 >nul 2>&1 && set"
        $imported = 0
        foreach ($line in $envLines) {
            if ($line -match '^(INCLUDE|LIB|LIBPATH)=(.*)$') {
                Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
                $imported++
            }
        }
        if ($imported -gt 0) {
            Write-Host "Imported $imported environment variable(s) (INCLUDE/LIB/LIBPATH)." -ForegroundColor Green
        } else {
            Write-Warning "vcvarsall.bat ran but no INCLUDE/LIB/LIBPATH came back -- bindgen will likely still fail to find stdbool.h."
        }
    } else {
        Write-Warning "vcvarsall.bat not found at $vcvarsall -- INCLUDE/LIB not set, bindgen may fail to find standard C headers (stdbool.h etc). Run this script from a 'Developer PowerShell for VS' instead, or set INCLUDE/LIB manually."
    }
}

# --- 4. Prefer Ninja for CMake if available, to avoid cmake's own VS-instance-detection bugs ---
$ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue
if ($ninja) {
    $env:CMAKE_GENERATOR = "Ninja"
    Write-Host "CMAKE_GENERATOR = Ninja ($($ninja.Source))" -ForegroundColor Green
} else {
    Write-Warning "ninja.exe not found on PATH -- letting CMake auto-detect a generator. If the build fails with 'could not find any instance of Visual Studio' for a generator like 'Visual Studio NN 20XX', install Ninja (winget install Ninja-build.Ninja) and re-run this script."
}

# --- 5. Warn about OneDrive-synced repo paths (observed to cause stale/locked build-cache dirs) ---
if ($RepoRoot -match "OneDrive") {
    Write-Warning "Repo path is under a OneDrive-synced folder ($RepoRoot). This caused spurious build-cache locks (cargo clean failing mid-cleanup, stale linked artifacts) on the machine this was first debugged on. Consider setting CARGO_TARGET_DIR to a path outside OneDrive, e.g.:`n  `$env:CARGO_TARGET_DIR = 'C:\meetily-build-target'"
}

# --- 6. Run cargo ---
Write-Host "`nRunning: cargo $CargoCommand (in $CrateDir)" -ForegroundColor Cyan
Push-Location $CrateDir
try {
    Invoke-Expression "cargo $CargoCommand"
} finally {
    Pop-Location
}
