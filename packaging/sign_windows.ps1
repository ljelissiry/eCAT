param(
    [string]$Path = "dist\standalone\eCAT Workbench\eCAT Workbench.exe",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [string]$CertificateThumbprint = ""
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Path)) {
    throw "Executable not found: $Path"
}

$signtool = Get-Command signtool.exe -ErrorAction SilentlyContinue
if ($null -eq $signtool) {
    throw "signtool.exe was not found. Install the Windows SDK and run this from a Developer PowerShell."
}

$arguments = @(
    "sign",
    "/fd", "SHA256",
    "/tr", $TimestampUrl,
    "/td", "SHA256"
)

if ($CertificateThumbprint) {
    $arguments += @("/sha1", $CertificateThumbprint)
} else {
    $arguments += "/a"
}

$arguments += $Path

& $signtool.Source @arguments
& $signtool.Source verify /pa /v $Path

Write-Host "Signed Windows artifact: $Path"

