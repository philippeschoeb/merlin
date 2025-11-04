window.BENCHMARK_DATA = {
  "lastUpdate": 1762269784576,
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
        "date": 1762269783122,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/benchmark_layer.py::test_quantum_layer_forward_benchmark[cpu-config0]",
            "value": 273.04535760361415,
            "unit": "iter/sec",
            "range": "stddev: 0.00004159580797538858",
            "extra": "mean: 3.6623951741077447 msec\nrounds: 224"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_quantum_layer_forward_benchmark[cpu-config1]",
            "value": 113.80076479045672,
            "unit": "iter/sec",
            "range": "stddev: 0.0006527960120681607",
            "extra": "mean: 8.787287166666383 msec\nrounds: 120"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_quantum_layer_forward_benchmark[cpu-config2]",
            "value": 52.066074996271745,
            "unit": "iter/sec",
            "range": "stddev: 0.0005327096982398748",
            "extra": "mean: 19.206364222223517 msec\nrounds: 54"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_quantum_layer_forward_benchmark[cpu-config3]",
            "value": 20.30716617713498,
            "unit": "iter/sec",
            "range": "stddev: 0.001601411942108099",
            "extra": "mean: 49.243700045452826 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-8-config0]",
            "value": 287.8019034899018,
            "unit": "iter/sec",
            "range": "stddev: 0.00004434275993501879",
            "extra": "mean: 3.474612182455865 msec\nrounds: 285"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-8-config1]",
            "value": 135.73341277512264,
            "unit": "iter/sec",
            "range": "stddev: 0.00007958921868195855",
            "extra": "mean: 7.367382721428787 msec\nrounds: 140"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-8-config2]",
            "value": 74.93697875024613,
            "unit": "iter/sec",
            "range": "stddev: 0.0005086859440884714",
            "extra": "mean: 13.344546533332391 msec\nrounds: 60"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-8-config3]",
            "value": 46.34413777357669,
            "unit": "iter/sec",
            "range": "stddev: 0.0005930118939718042",
            "extra": "mean: 21.577702122449548 msec\nrounds: 49"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-16-config0]",
            "value": 281.8877444992054,
            "unit": "iter/sec",
            "range": "stddev: 0.000057402215884549776",
            "extra": "mean: 3.5475114456521504 msec\nrounds: 276"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-16-config1]",
            "value": 131.52189801460017,
            "unit": "iter/sec",
            "range": "stddev: 0.0002582844647331754",
            "extra": "mean: 7.603296600000333 msec\nrounds: 135"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-16-config2]",
            "value": 73.85030455617292,
            "unit": "iter/sec",
            "range": "stddev: 0.00018327784359940458",
            "extra": "mean: 13.540905565790426 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-16-config3]",
            "value": 44.35262829659977,
            "unit": "iter/sec",
            "range": "stddev: 0.0012857193934713402",
            "extra": "mean: 22.54657814893607 msec\nrounds: 47"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-32-config0]",
            "value": 275.8318982446465,
            "unit": "iter/sec",
            "range": "stddev: 0.0000945419271451264",
            "extra": "mean: 3.625396505494298 msec\nrounds: 273"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-32-config1]",
            "value": 130.13110124802353,
            "unit": "iter/sec",
            "range": "stddev: 0.00007438198010051362",
            "extra": "mean: 7.684558037313838 msec\nrounds: 134"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-32-config2]",
            "value": 71.49871845746834,
            "unit": "iter/sec",
            "range": "stddev: 0.00019526815662794213",
            "extra": "mean: 13.986264671231263 msec\nrounds: 73"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-32-config3]",
            "value": 41.90372534020206,
            "unit": "iter/sec",
            "range": "stddev: 0.000986431310140997",
            "extra": "mean: 23.86422667391362 msec\nrounds: 46"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-64-config0]",
            "value": 260.2141311600684,
            "unit": "iter/sec",
            "range": "stddev: 0.00038851921243238",
            "extra": "mean: 3.842988832089441 msec\nrounds: 268"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-64-config1]",
            "value": 122.42206926259,
            "unit": "iter/sec",
            "range": "stddev: 0.00012531869552918532",
            "extra": "mean: 8.168461830644633 msec\nrounds: 124"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-64-config2]",
            "value": 65.59219945602553,
            "unit": "iter/sec",
            "range": "stddev: 0.00033176405439161054",
            "extra": "mean: 15.245715318182343 msec\nrounds: 66"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-64-config3]",
            "value": 36.89848011538589,
            "unit": "iter/sec",
            "range": "stddev: 0.0008095738762156285",
            "extra": "mean: 27.101387289473234 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_gradient_computation_benchmark[cpu-config0]",
            "value": 128.58344457581342,
            "unit": "iter/sec",
            "range": "stddev: 0.00004715233047218421",
            "extra": "mean: 7.777050951612944 msec\nrounds: 124"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_gradient_computation_benchmark[cpu-config1]",
            "value": 52.5174136912074,
            "unit": "iter/sec",
            "range": "stddev: 0.0002475305368901661",
            "extra": "mean: 19.041303250000343 msec\nrounds: 52"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_multiple_circuit_types_benchmark[cpu-config0]",
            "value": 29.104301638763893,
            "unit": "iter/sec",
            "range": "stddev: 0.0016777795974233388",
            "extra": "mean: 34.35918210344908 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_multiple_circuit_types_benchmark[cpu-config1]",
            "value": 12.15913972612364,
            "unit": "iter/sec",
            "range": "stddev: 0.0023432941933802915",
            "extra": "mean: 82.24266046153926 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_output_mapping_strategies_benchmark[cpu-config0]",
            "value": 19.490917265381434,
            "unit": "iter/sec",
            "range": "stddev: 0.001988754465031728",
            "extra": "mean: 51.30594863157817 msec\nrounds: 19"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_output_mapping_strategies_benchmark[cpu-config1]",
            "value": 7.038277428583748,
            "unit": "iter/sec",
            "range": "stddev: 0.05183173341224454",
            "extra": "mean: 142.08021922222258 msec\nrounds: 9"
          }
        ]
      },
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
        "date": 1762269783122,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/benchmark_layer.py::test_quantum_layer_forward_benchmark[cpu-config0]",
            "value": 273.04535760361415,
            "unit": "iter/sec",
            "range": "stddev: 0.00004159580797538858",
            "extra": "mean: 3.6623951741077447 msec\nrounds: 224"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_quantum_layer_forward_benchmark[cpu-config1]",
            "value": 113.80076479045672,
            "unit": "iter/sec",
            "range": "stddev: 0.0006527960120681607",
            "extra": "mean: 8.787287166666383 msec\nrounds: 120"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_quantum_layer_forward_benchmark[cpu-config2]",
            "value": 52.066074996271745,
            "unit": "iter/sec",
            "range": "stddev: 0.0005327096982398748",
            "extra": "mean: 19.206364222223517 msec\nrounds: 54"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_quantum_layer_forward_benchmark[cpu-config3]",
            "value": 20.30716617713498,
            "unit": "iter/sec",
            "range": "stddev: 0.001601411942108099",
            "extra": "mean: 49.243700045452826 msec\nrounds: 22"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-8-config0]",
            "value": 287.8019034899018,
            "unit": "iter/sec",
            "range": "stddev: 0.00004434275993501879",
            "extra": "mean: 3.474612182455865 msec\nrounds: 285"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-8-config1]",
            "value": 135.73341277512264,
            "unit": "iter/sec",
            "range": "stddev: 0.00007958921868195855",
            "extra": "mean: 7.367382721428787 msec\nrounds: 140"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-8-config2]",
            "value": 74.93697875024613,
            "unit": "iter/sec",
            "range": "stddev: 0.0005086859440884714",
            "extra": "mean: 13.344546533332391 msec\nrounds: 60"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-8-config3]",
            "value": 46.34413777357669,
            "unit": "iter/sec",
            "range": "stddev: 0.0005930118939718042",
            "extra": "mean: 21.577702122449548 msec\nrounds: 49"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-16-config0]",
            "value": 281.8877444992054,
            "unit": "iter/sec",
            "range": "stddev: 0.000057402215884549776",
            "extra": "mean: 3.5475114456521504 msec\nrounds: 276"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-16-config1]",
            "value": 131.52189801460017,
            "unit": "iter/sec",
            "range": "stddev: 0.0002582844647331754",
            "extra": "mean: 7.603296600000333 msec\nrounds: 135"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-16-config2]",
            "value": 73.85030455617292,
            "unit": "iter/sec",
            "range": "stddev: 0.00018327784359940458",
            "extra": "mean: 13.540905565790426 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-16-config3]",
            "value": 44.35262829659977,
            "unit": "iter/sec",
            "range": "stddev: 0.0012857193934713402",
            "extra": "mean: 22.54657814893607 msec\nrounds: 47"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-32-config0]",
            "value": 275.8318982446465,
            "unit": "iter/sec",
            "range": "stddev: 0.0000945419271451264",
            "extra": "mean: 3.625396505494298 msec\nrounds: 273"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-32-config1]",
            "value": 130.13110124802353,
            "unit": "iter/sec",
            "range": "stddev: 0.00007438198010051362",
            "extra": "mean: 7.684558037313838 msec\nrounds: 134"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-32-config2]",
            "value": 71.49871845746834,
            "unit": "iter/sec",
            "range": "stddev: 0.00019526815662794213",
            "extra": "mean: 13.986264671231263 msec\nrounds: 73"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-32-config3]",
            "value": 41.90372534020206,
            "unit": "iter/sec",
            "range": "stddev: 0.000986431310140997",
            "extra": "mean: 23.86422667391362 msec\nrounds: 46"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-64-config0]",
            "value": 260.2141311600684,
            "unit": "iter/sec",
            "range": "stddev: 0.00038851921243238",
            "extra": "mean: 3.842988832089441 msec\nrounds: 268"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-64-config1]",
            "value": 122.42206926259,
            "unit": "iter/sec",
            "range": "stddev: 0.00012531869552918532",
            "extra": "mean: 8.168461830644633 msec\nrounds: 124"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-64-config2]",
            "value": 65.59219945602553,
            "unit": "iter/sec",
            "range": "stddev: 0.00033176405439161054",
            "extra": "mean: 15.245715318182343 msec\nrounds: 66"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_batched_computation_benchmark[cpu-64-config3]",
            "value": 36.89848011538589,
            "unit": "iter/sec",
            "range": "stddev: 0.0008095738762156285",
            "extra": "mean: 27.101387289473234 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_gradient_computation_benchmark[cpu-config0]",
            "value": 128.58344457581342,
            "unit": "iter/sec",
            "range": "stddev: 0.00004715233047218421",
            "extra": "mean: 7.777050951612944 msec\nrounds: 124"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_gradient_computation_benchmark[cpu-config1]",
            "value": 52.5174136912074,
            "unit": "iter/sec",
            "range": "stddev: 0.0002475305368901661",
            "extra": "mean: 19.041303250000343 msec\nrounds: 52"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_multiple_circuit_types_benchmark[cpu-config0]",
            "value": 29.104301638763893,
            "unit": "iter/sec",
            "range": "stddev: 0.0016777795974233388",
            "extra": "mean: 34.35918210344908 msec\nrounds: 29"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_multiple_circuit_types_benchmark[cpu-config1]",
            "value": 12.15913972612364,
            "unit": "iter/sec",
            "range": "stddev: 0.0023432941933802915",
            "extra": "mean: 82.24266046153926 msec\nrounds: 13"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_output_mapping_strategies_benchmark[cpu-config0]",
            "value": 19.490917265381434,
            "unit": "iter/sec",
            "range": "stddev: 0.001988754465031728",
            "extra": "mean: 51.30594863157817 msec\nrounds: 19"
          },
          {
            "name": "benchmarks/benchmark_layer.py::test_output_mapping_strategies_benchmark[cpu-config1]",
            "value": 7.038277428583748,
            "unit": "iter/sec",
            "range": "stddev: 0.05183173341224454",
            "extra": "mean: 142.08021922222258 msec\nrounds: 9"
          }
        ]
      }
    ]
  }
}