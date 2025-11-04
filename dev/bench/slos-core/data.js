window.BENCHMARK_DATA = {
  "lastUpdate": 1762269783395,
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
        "date": 1762269782882,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/benchmark_slos_core.py::test_build_graph_benchmark[dtype_pair0-cpu-config0]",
            "value": 17212.808665210647,
            "unit": "iter/sec",
            "range": "stddev: 0.00003354531277827989",
            "extra": "mean: 58.096271180956755 usec\nrounds: 2998"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_build_graph_benchmark[dtype_pair0-cpu-config1]",
            "value": 6006.6806454836515,
            "unit": "iter/sec",
            "range": "stddev: 0.00004485764635147386",
            "extra": "mean: 166.4812995763122 usec\nrounds: 4957"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_build_graph_benchmark[dtype_pair0-cpu-config2]",
            "value": 1399.0483020006325,
            "unit": "iter/sec",
            "range": "stddev: 0.00014170189152215289",
            "extra": "mean: 714.7716047902026 usec\nrounds: 668"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_build_graph_benchmark[dtype_pair0-cpu-config3]",
            "value": 287.5674948561552,
            "unit": "iter/sec",
            "range": "stddev: 0.005323244792295834",
            "extra": "mean: 3.4774444882938256 msec\nrounds: 299"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_build_graph_benchmark[dtype_pair1-cpu-config0]",
            "value": 16927.52225949686,
            "unit": "iter/sec",
            "range": "stddev: 0.000029491861088638357",
            "extra": "mean: 59.075391227973086 usec\nrounds: 9097"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_build_graph_benchmark[dtype_pair1-cpu-config1]",
            "value": 5885.046744556446,
            "unit": "iter/sec",
            "range": "stddev: 0.00004607968095510374",
            "extra": "mean: 169.92218471756075 usec\nrounds: 4986"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_build_graph_benchmark[dtype_pair1-cpu-config2]",
            "value": 1464.3594015627975,
            "unit": "iter/sec",
            "range": "stddev: 0.0000643525571582429",
            "extra": "mean: 682.8924640581931 usec\nrounds: 1099"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_build_graph_benchmark[dtype_pair1-cpu-config3]",
            "value": 262.35664177202676,
            "unit": "iter/sec",
            "range": "stddev: 0.007544558601789866",
            "extra": "mean: 3.8116054285713266 msec\nrounds: 287"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_benchmark[dtype_pair0-cpu-config0]",
            "value": 9265.517012988186,
            "unit": "iter/sec",
            "range": "stddev: 0.000008223770121465799",
            "extra": "mean: 107.92705885685854 usec\nrounds: 1784"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_benchmark[dtype_pair0-cpu-config1]",
            "value": 6453.612840863904,
            "unit": "iter/sec",
            "range": "stddev: 0.000016104830388042182",
            "extra": "mean: 154.95196638820005 usec\nrounds: 5147"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_benchmark[dtype_pair0-cpu-config2]",
            "value": 4225.577090321721,
            "unit": "iter/sec",
            "range": "stddev: 0.000012393045142987865",
            "extra": "mean: 236.6540660896719 usec\nrounds: 3465"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_benchmark[dtype_pair0-cpu-config3]",
            "value": 1992.7277808138026,
            "unit": "iter/sec",
            "range": "stddev: 0.000024242511943155022",
            "extra": "mean: 501.8246895677912 usec\nrounds: 1572"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_benchmark[dtype_pair1-cpu-config0]",
            "value": 9243.441661100409,
            "unit": "iter/sec",
            "range": "stddev: 0.00000843199895807604",
            "extra": "mean: 108.18481217968248 usec\nrounds: 6043"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_benchmark[dtype_pair1-cpu-config1]",
            "value": 6359.936296273933,
            "unit": "iter/sec",
            "range": "stddev: 0.00001030132853952883",
            "extra": "mean: 157.23427930966312 usec\nrounds: 5041"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_benchmark[dtype_pair1-cpu-config2]",
            "value": 4011.455788113923,
            "unit": "iter/sec",
            "range": "stddev: 0.000012998119188213647",
            "extra": "mean: 249.28605793513498 usec\nrounds: 3245"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_benchmark[dtype_pair1-cpu-config3]",
            "value": 1828.0530952921545,
            "unit": "iter/sec",
            "range": "stddev: 0.00001837360314422353",
            "extra": "mean: 547.0300630629017 usec\nrounds: 1554"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[8-dtype_pair0-cpu-config0]",
            "value": 9530.092705961424,
            "unit": "iter/sec",
            "range": "stddev: 0.000008668755125261266",
            "extra": "mean: 104.93077358780184 usec\nrounds: 6603"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[8-dtype_pair0-cpu-config1]",
            "value": 6697.0104390045835,
            "unit": "iter/sec",
            "range": "stddev: 0.000010487982263963775",
            "extra": "mean: 149.3203585551878 usec\nrounds: 5260"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[8-dtype_pair1-cpu-config0]",
            "value": 9446.466230312359,
            "unit": "iter/sec",
            "range": "stddev: 0.000010018008359791134",
            "extra": "mean: 105.8596914040875 usec\nrounds: 6724"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[8-dtype_pair1-cpu-config1]",
            "value": 6688.716583168545,
            "unit": "iter/sec",
            "range": "stddev: 0.000010157316693868629",
            "extra": "mean: 149.50551239028354 usec\nrounds: 5125"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[16-dtype_pair0-cpu-config0]",
            "value": 9316.595613481813,
            "unit": "iter/sec",
            "range": "stddev: 0.000008679006962631282",
            "extra": "mean: 107.33534452787936 usec\nrounds: 6551"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[16-dtype_pair0-cpu-config1]",
            "value": 6474.753972743674,
            "unit": "iter/sec",
            "range": "stddev: 0.000010744508078301568",
            "extra": "mean: 154.4460228465253 usec\nrounds: 4202"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[16-dtype_pair1-cpu-config0]",
            "value": 9293.318012674881,
            "unit": "iter/sec",
            "range": "stddev: 0.00000845202642466911",
            "extra": "mean: 107.60419460908683 usec\nrounds: 6752"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[16-dtype_pair1-cpu-config1]",
            "value": 6135.144574135159,
            "unit": "iter/sec",
            "range": "stddev: 0.000030166806858584285",
            "extra": "mean: 162.99534394280593 usec\nrounds: 5027"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[32-dtype_pair0-cpu-config0]",
            "value": 9063.007330756174,
            "unit": "iter/sec",
            "range": "stddev: 0.000008819985547742039",
            "extra": "mean: 110.33865068236292 usec\nrounds: 5130"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[32-dtype_pair0-cpu-config1]",
            "value": 5999.539700467272,
            "unit": "iter/sec",
            "range": "stddev: 0.000010302929291974061",
            "extra": "mean: 166.6794537457791 usec\nrounds: 4659"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[32-dtype_pair1-cpu-config0]",
            "value": 8892.946670758522,
            "unit": "iter/sec",
            "range": "stddev: 0.000009282652862891746",
            "extra": "mean: 112.44866713169046 usec\nrounds: 6453"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[32-dtype_pair1-cpu-config1]",
            "value": 5787.492626901075,
            "unit": "iter/sec",
            "range": "stddev: 0.000009973505148567016",
            "extra": "mean: 172.78639723044486 usec\nrounds: 4116"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[64-dtype_pair0-cpu-config0]",
            "value": 8554.671136618683,
            "unit": "iter/sec",
            "range": "stddev: 0.000008555312060014809",
            "extra": "mean: 116.89520076574911 usec\nrounds: 6007"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[64-dtype_pair0-cpu-config1]",
            "value": 5254.250228449459,
            "unit": "iter/sec",
            "range": "stddev: 0.000010617553241535356",
            "extra": "mean: 190.32211191340656 usec\nrounds: 4155"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[64-dtype_pair1-cpu-config0]",
            "value": 8358.306262561404,
            "unit": "iter/sec",
            "range": "stddev: 0.000011555324008928073",
            "extra": "mean: 119.6414642616302 usec\nrounds: 5960"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[64-dtype_pair1-cpu-config1]",
            "value": 5003.6044525501275,
            "unit": "iter/sec",
            "range": "stddev: 0.000011190476566837911",
            "extra": "mean: 199.85592575974746 usec\nrounds: 4014"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[128-dtype_pair0-cpu-config0]",
            "value": 7929.72436876648,
            "unit": "iter/sec",
            "range": "stddev: 0.000014451725391084425",
            "extra": "mean: 126.10778805109419 usec\nrounds: 5624"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[128-dtype_pair0-cpu-config1]",
            "value": 4233.0843382466655,
            "unit": "iter/sec",
            "range": "stddev: 0.000012843416739383276",
            "extra": "mean: 236.2343672118278 usec\nrounds: 3178"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[128-dtype_pair1-cpu-config0]",
            "value": 7612.753064681642,
            "unit": "iter/sec",
            "range": "stddev: 0.000010824720937966643",
            "extra": "mean: 131.35852319174353 usec\nrounds: 5627"
          },
          {
            "name": "benchmarks/benchmark_slos_core.py::test_compute_batched_benchmark[128-dtype_pair1-cpu-config1]",
            "value": 3911.3266625248366,
            "unit": "iter/sec",
            "range": "stddev: 0.000011042879159522231",
            "extra": "mean: 255.66772767439497 usec\nrounds: 3169"
          }
        ]
      }
    ]
  }
}