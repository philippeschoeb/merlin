window.BENCHMARK_DATA = {
  "lastUpdate": 1762269793808,
  "repoUrl": "https://github.com/philippeschoeb/merlin",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "90058728+ben9871@users.noreply.github.com",
            "name": "Benjamin Stott",
            "username": "ben9871"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a77a37b7a1826add10f9d446ef120c6c5789203c",
          "message": "Merge pull request #70 from merlinquantum/bugfix/packaging\n\nfix docs dependency for pypi",
          "timestamp": "2025-11-04T11:56:57+01:00",
          "tree_id": "64a91c9c4c536a58ea3ef7a5e70eacbd91270b92",
          "url": "https://github.com/philippeschoeb/merlin/commit/a77a37b7a1826add10f9d446ef120c6c5789203c"
        },
        "date": 1762269793238,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-64-config0]",
            "value": 106.60948270473457,
            "unit": "iter/sec",
            "range": "stddev: 0.0006773423416666129",
            "extra": "mean: 9.380028630000936 msec\nrounds: 100"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-64-config1]",
            "value": 45.64475883514248,
            "unit": "iter/sec",
            "range": "stddev: 0.0005073324096086518",
            "extra": "mean: 21.9083203750019 msec\nrounds: 48"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-64-config2]",
            "value": 18.973292511371366,
            "unit": "iter/sec",
            "range": "stddev: 0.0014634187767121058",
            "extra": "mean: 52.70566504999934 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-128-config0]",
            "value": 100.73873179092143,
            "unit": "iter/sec",
            "range": "stddev: 0.00008226682641972385",
            "extra": "mean: 9.926668543688377 msec\nrounds: 103"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-128-config1]",
            "value": 43.055597428754666,
            "unit": "iter/sec",
            "range": "stddev: 0.0001761516157035085",
            "extra": "mean: 23.225783863636515 msec\nrounds: 44"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-128-config2]",
            "value": 16.11368587818542,
            "unit": "iter/sec",
            "range": "stddev: 0.003970584454067046",
            "extra": "mean: 62.05904766666651 msec\nrounds: 18"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-256-config0]",
            "value": 85.57898306367649,
            "unit": "iter/sec",
            "range": "stddev: 0.00011712100434843015",
            "extra": "mean: 11.685111977270552 msec\nrounds: 88"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-256-config1]",
            "value": 34.46673545135505,
            "unit": "iter/sec",
            "range": "stddev: 0.0001817354662016238",
            "extra": "mean: 29.013481750000935 msec\nrounds: 36"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-256-config2]",
            "value": 13.364640104604064,
            "unit": "iter/sec",
            "range": "stddev: 0.0028086413038359704",
            "extra": "mean: 74.82431192857219 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-512-config0]",
            "value": 66.04936457346712,
            "unit": "iter/sec",
            "range": "stddev: 0.0012460948111192858",
            "extra": "mean: 15.140191074627126 msec\nrounds: 67"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-512-config1]",
            "value": 25.39779015312493,
            "unit": "iter/sec",
            "range": "stddev: 0.0003010431779939007",
            "extra": "mean: 39.37350430769507 msec\nrounds: 26"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_large_batch_robustness_benchmark[cpu-512-config2]",
            "value": 9.582728987749093,
            "unit": "iter/sec",
            "range": "stddev: 0.003925607814188347",
            "extra": "mean: 104.35440690000064 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_extreme_values_robustness_benchmark[cpu-config0]",
            "value": 18.26387222521943,
            "unit": "iter/sec",
            "range": "stddev: 0.0011827631792286615",
            "extra": "mean: 54.75290166666645 msec\nrounds: 18"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_extreme_values_robustness_benchmark[cpu-config1]",
            "value": 7.99587393714748,
            "unit": "iter/sec",
            "range": "stddev: 0.002675516398673204",
            "extra": "mean: 125.06450299999963 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_extreme_values_robustness_benchmark[cpu-config2]",
            "value": 3.395775080962453,
            "unit": "iter/sec",
            "range": "stddev: 0.0026573242278920446",
            "extra": "mean: 294.48357920000205 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_numerical_stability_benchmark[cpu-config0]",
            "value": 8.317406347321617,
            "unit": "iter/sec",
            "range": "stddev: 0.000505357926450578",
            "extra": "mean: 120.2297877777754 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_numerical_stability_benchmark[cpu-config1]",
            "value": 3.5994577831506787,
            "unit": "iter/sec",
            "range": "stddev: 0.0041488695804521875",
            "extra": "mean: 277.81962180000335 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_memory_efficiency_benchmark[cpu-config0]",
            "value": 1.7227726236312395,
            "unit": "iter/sec",
            "range": "stddev: 0.0024320176053843107",
            "extra": "mean: 580.4596534000012 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_memory_efficiency_benchmark[cpu-config1]",
            "value": 0.7466922133051019,
            "unit": "iter/sec",
            "range": "stddev: 0.009538874515393533",
            "extra": "mean: 1.339239893200005 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_hybrid_model_stress_benchmark[cpu-config0]",
            "value": 18.401323612114563,
            "unit": "iter/sec",
            "range": "stddev: 0.0004217856188970634",
            "extra": "mean: 54.34391683333296 msec\nrounds: 18"
          },
          {
            "name": "benchmarks/benchmark_robustness.py::test_hybrid_model_stress_benchmark[cpu-config1]",
            "value": 7.803659398875414,
            "unit": "iter/sec",
            "range": "stddev: 0.00171504049544198",
            "extra": "mean: 128.14500850000064 msec\nrounds: 8"
          }
        ]
      }
    ]
  }
}