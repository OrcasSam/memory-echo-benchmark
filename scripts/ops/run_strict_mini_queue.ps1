param(
    [Parameter(Mandatory = $true)][string]$Starts,
    [Parameter(Mandatory = $true)][string]$Lane
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$python = 'python'
$env:HF_HOME = (Join-Path $root '.cache/hf')
$env:TRANSFORMERS_CACHE = (Join-Path $root '.cache/hf/transformers')
$env:MPLCONFIGDIR = (Join-Path $root '.cache/matplotlib')
$env:MEM0_DIR = (Join-Path $root '.cache/mem0')

foreach ($startText in $Starts.Split(',')) {
    $start = [int]$startText
    $end = $start + 5
    $output = Join-Path $root "results_memory_echo_solution_strict_mini_${start}_${end}"
    $stdout = Join-Path $root "logs\strict_mini_${start}_${end}_stdout.log"
    $stderr = Join-Path $root "logs\strict_mini_${start}_${end}_stderr.log"
    $arguments = @(
        '-m', 'scripts.legacy.run_memory_echo_solution_validation', '--max_samples', '200',
        '--sample_start', "$start", '--sample_end', "$end",
        '--frameworks', 'amem,mem0', '--output_dir', $output
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $root -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -WindowStyle Hidden -PassThru -Wait
    if ($process.ExitCode -ne 0) {
        throw "Strict mini-shard $start-$end failed with exit code $($process.ExitCode)"
    }
}
