#Requires -Version 5.1

param(
    [string]$ScriptDir = $PSScriptRoot
)

#=============================================================================
#region SCRIPT DETAILS
#=============================================================================
<#
.SYNOPSIS
Checks for Windows OS, verifies prerequisites,
and installs required Python packages for Assignment 2.

.DESCRIPTION
- Verifies Python 3.8+ is installed and on PATH
- Upgrades pip to the latest version
- Detects imports dynamically from the assignment script
- Installs mapped packages
- Offers to launch the assignment script after setup

.NOTES
Run from the folder containing the .py files.
#>
#=============================================================================
#endregion
#=============================================================================
#region Prerequisites
#=============================================================================

$OS = (Get-CimInstance -ClassName Win32_OperatingSystem).Caption
$Windows = ($OS -match 'Windows')
if (!$Windows) {
    Write-Output 'OS is not Windows. This script is only intended for Windows devices'
    exit 666
}

#=============================================================================
#endregion
#=============================================================================
#region VARIABLES
#=============================================================================

$VerbosePreference = 'Continue'
$EnableLogging = $False
$LogFileName = 'Install-Assignment_02_Attention_Under_The_Hood_Windows.log'
$LogFile = Join-Path -Path $PSScriptRoot -ChildPath $LogFileName
$AssignmentFile = Join-Path $ScriptDir 'scripts\Assignment_02_Attention_Under_The_Hood.py'

#=============================================================================
#endregion
#=============================================================================
#region FUNCTIONS
#=============================================================================

function Write-Log {
    [CmdletBinding()]
    param (
        [AllowEmptyString()]
        [Parameter(Mandatory = $true,Position = 0)][String]$LogText,
        [Alias('ForegroundColor')]
        [Parameter(Mandatory = $false,Position = 1)][System.ConsoleColor]$Color = [System.ConsoleColor]::White
    )
    if ($EnableLogging) {
        $CurrentTime = Get-Date
        Add-Content $LogFile "$CurrentTime - $LogText"
    }
    Write-Host $LogText -ForegroundColor $Color
}

#=============================================================================
#endregion
#=============================================================================
#region EXECUTION
#=============================================================================

if ($EnableLogging) {
    if (!(Test-Path $LogFile)) { New-Item -ItemType File -Path $LogFile -Force }
}
Write-Log "Computer Name is: $((Get-CimInstance -ClassName Win32_ComputerSystem).Name)" -ForegroundColor DarkYellow
Write-Log "Current Time Zone is $((Get-TimeZone).DisplayName)" -ForegroundColor DarkYellow

Clear-Host
Write-Log 'Assignment 2 Attention Under the Hood - Windows Installer' -ForegroundColor Cyan
Write-Log ''
Write-Log '  This script will verify your Python environment, dynamically parse'
Write-Log '  required dependencies, and prepare your system to run:'
Write-Log '    * Assignment_02_Attention_Under_The_Hood.py'

#=============================================================================
#1. Locate Python
#=============================================================================
Write-Log ''
Write-Log 'Checking for Python...' -ForegroundColor Cyan

$pythonExe = $null
$candidates = @('py', 'python', 'python3')

foreach ($cmd in $candidates) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match 'Python (\d+)\.(\d+)') {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 8) {
                $pythonExe = $cmd
                Write-Log "Found: $ver  (using '$cmd')" -ForegroundColor Green
                break
            } else {
                Write-Log "Found $ver but Python 3.8+ is required - skipping." -ForegroundColor Yellow
            }
        }
    } catch { }
}

if (-not $pythonExe) {
    Write-Log 'Python 3.8+ was not found on this machine.' -ForegroundColor Red
    Write-Log ''
    Write-Log "  IMPORTANT: During install, tick 'Add Python to PATH'."
    $open = Read-Host '  Open the Python download page now? [Y/N]'
    if ($open -match '^[Yy]') { Start-Process 'https://www.python.org/downloads/' }
    Write-Log '  Re-run this installer after Python is installed.'
    exit 1
}

#=============================================================================
#2. Verify & Upgrade pip
#=============================================================================
Write-Log ''
Write-Log 'Checking pip...' -ForegroundColor Cyan

& $pythonExe -m ensurepip --upgrade 2>$null | Out-Null
& $pythonExe -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -eq 0) { Write-Log 'pip is ready and up to date.' -ForegroundColor Green }

#=============================================================================
#3. Check Application File
#=============================================================================
Write-Log ''
Write-Log 'Checking application files...' -ForegroundColor Cyan

if (Test-Path $AssignmentFile) {
    Write-Log 'Assignment_02_Attention_Under_The_Hood.py found' -ForegroundColor Green
} else {
    Write-Log "Assignment_02_Attention_Under_The_Hood.py not found in $ScriptDir" -ForegroundColor Red
    Write-Log 'This file must be present to detect required packages.' -ForegroundColor Yellow
    exit 1
}

#=============================================================================
#4. Detect & install required packages
#=============================================================================
Write-Log ''
Write-Log 'Detecting & installing packages...' -ForegroundColor Cyan

$importJson = & $pythonExe -c @"
import ast
import json
from pathlib import Path

source = Path(r'''$AssignmentFile''').read_text(encoding='utf-8')
magic_prefixes = (chr(33), chr(37))
source = '\n'.join('' if line.lstrip()[:1] in magic_prefixes else line for line in source.splitlines())
tree = ast.parse(source)
imports = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            imports.add(alias.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom) and node.module:
        imports.add(node.module.split('.')[0])
print(json.dumps(sorted(imports)))
"@

if ($LASTEXITCODE -ne 0) {
    Write-Log 'Could not parse imports from full assignment script.' -ForegroundColor Red
    exit 1
}

$detectedImports = $importJson | ConvertFrom-Json
$packageMap = @{
    'torch'      = 'torch'
    'numpy'      = 'numpy'
    'matplotlib' = 'matplotlib'
    'seaborn'    = 'seaborn'
}
$standardLibrary = @('math', 'pathlib')

$packages = [System.Collections.Generic.List[string]]::new()
foreach ($importName in $detectedImports) {
    if ($standardLibrary -contains $importName) { continue }
    if ($packageMap.ContainsKey($importName)) {
        $pkg = $packageMap[$importName]
        if (-not $packages.Contains($pkg)) { $packages.Add($pkg) }
    } else {
        Write-Log "Unmapped third-party import detected: $importName" -ForegroundColor Red
        exit 1
    }
}

Write-Log "Packages needed: $($packages -join ', ')" -ForegroundColor Gray

foreach ($pkg in $packages) {
    Write-Log "  Installing $pkg..."
    & $pythonExe -m pip install --quiet $pkg
    if ($LASTEXITCODE -eq 0) { Write-Log '   done' -ForegroundColor Green }
    else { Write-Log '   FAILED' -ForegroundColor Red; exit 1 }
}

#=============================================================================
#5. Verify Imports
#=============================================================================
Write-Log ''
Write-Log 'Verifying imports...' -ForegroundColor Cyan

$allGood = $true
$verifyImports = @('torch', 'numpy', 'matplotlib', 'seaborn')
foreach ($importName in $verifyImports) {
    & $pythonExe -c "import $importName" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Import verification failed: $importName" -ForegroundColor Red
        $allGood = $false
    } else { Write-Log "$importName imports successfully" }
}

#=============================================================================
#6. Summary & launch prompt
#=============================================================================
Write-Log ''
Write-Log 'Setup Complete' -ForegroundColor Cyan

if ($allGood) {
    Write-Log ''
    Write-Log '  Everything is ready. Would you like to launch the assignment script?' -ForegroundColor White
    Write-Log ''
    Write-Log '  [1] Assignment_02_Attention_Under_The_Hood.py' -ForegroundColor Gray
    Write-Log '  [2] Exit' -ForegroundColor Gray
    Write-Log ''

    $choice = Read-Host '  Enter choice [1/2]'

    switch ($choice) {
        '1' {
            Write-Log ''
            Write-Log 'Launching Assignment_02_Attention_Under_The_Hood.py...'
            Start-Process $pythonExe -ArgumentList "`"$AssignmentFile`""
        }
        default {
            Write-Log ''
            Write-Log '  To run manually:' -ForegroundColor White
            Write-Log "    $pythonExe `"$AssignmentFile`"" -ForegroundColor Cyan
        }
    }
} else {
    Write-Log ''
    Write-Log 'One or more packages could not be verified.' -ForegroundColor Red
}

Write-Log ''
#=============================================================================
#endregion
#=============================================================================
