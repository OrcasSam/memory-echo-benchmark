param(
    [Parameter(Mandatory = $true)][int]$WaitPid,
    [Parameter(Mandatory = $true)][string]$Ranges
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = 'python'
$env:HF_HOME = (Join-Path $root '.cache/hf')
$env:TRANSFORMERS_CACHE = (Join-Path $root '.cache/hf/transformers')
$env:MPLCONFIGDIR = (Join-Path $root '.cache/matplotlib')
$env:MEM0_DIR = (Join-Path $root '.cache/mem0')

Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
foreach ($range in $Ranges.Split(',')) {
    $bounds = $range.Split('-')
    $start = [int]$bounds[0]
    $end = [int]$bounds[1]
    $output = Join-Path $root "results_memory_echo_solution_strict_shard_${start}_${end}"
    $stdout = Join-Path $root "logs\strict_${start}_${end}_retry_stdout.log"
    $stderr = Join-Path $root "logs\strict_${start}_${end}_retry_stderr.log"
    $arguments = @(
        '-m', 'scripts.legacy.run_memory_echo_solution_validation', '--max_samples', '200',
        '--sample_start', "$start", '--sample_end', "$end",
        '--frameworks', 'amem,mem0', '--output_dir', $output
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $root -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "Strict solution shard $range failed with exit code $($process.ExitCode)"
    }
}
