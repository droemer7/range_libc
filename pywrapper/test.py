import range_libc
import numpy as np
import itertools, time

####################################################################################################
#
#                                              WARNING
#
#
#                    This file uses range_libc in it's native coordinate space.
#                      Use this method at your own peril since the coordinate
#                      conversions are nontrivial from ROS's coordinate space.
#                       Ignore this warning if you intend to use range_libc's
#                                   left handed coordinate space
#
#
####################################################################################################

# print(range_libc.USE_CACHED_TRIG)
# print(range_libc.USE_CACHED_TRIG)
# print(range_libc.USE_ALTERNATE_MOD)
# print(range_libc.USE_CACHED_CONSTANT)S
# print(range_libc.USE_FAST_ROUND)
# print(range_libc.NO_INLINE)
# print(range_libc.USE_LRU_CACHE)
# print(range_libc.LRU_CACHE_SIZE)

# testMap = range_libc.PyOMap("../maps/basement_hallways_5cm.png".encode('utf8'),1)
testMap = range_libc.PyOMap("../maps/test.png".encode('utf8'),1)
# testMap = range_libc.PyOMap("/home/racecar/racecar-ws/src/TA_examples/lab5/maps/basement.png".encode('utf8'),1)

if testMap.error():
  exit()
# testMap.save("./loaded_map.png".encode('utf-8'))

print("Initializing: bl")
bl = range_libc.PyBresenhamsLine(testMap, 500)
print("Initializing: rm")
rm = range_libc.PyRayMarching(testMap, 500)
print("Initializing: cddt")
cddt = range_libc.PyCDDTCast(testMap, 500, 108)
cddt.prune()
print("Initializing: glt")
glt = range_libc.PyGiantLUTCast(testMap, 500, 108)
print("")

# For testing / debugging
def fixed_scan(num_ranges, print_sample=True):
  ranges_np = np.zeros(num_ranges, dtype=np.float32)
  queries = np.zeros((num_ranges, 3), dtype=np.float32)
  queries[:,0] = testMap.width() / 2.0
  queries[:,1] = testMap.height() / 2.0
  queries[:,2] = np.linspace(0, 2.0 * np.pi, num_ranges)
  queries_deg = np.copy(queries)
  queries_deg[:,2] *= 180.0 / np.pi
  print("Points (x, y, th): \n", queries_deg)
  print()

  if print_sample:
    sample = np.arange(0, num_ranges)

  def scan(obj, name):
    print("Running " + name + ":")
    print("--------------------")
    obj.calc_range_many(queries, ranges_np)
    ranges_slow = np.array([obj.calc_range(*q) for q in queries],
                           dtype=np.float32)

    if print_sample:
      print("Numpy sample:", [ranges_np[int(s)] for s in sample])
      print("Slow sample: ", [ranges_slow[int(s)] for s in sample])
    print("Diff:", np.linalg.norm(ranges_np - ranges_slow))
    print("")

    if range_libc.MAKE_TRACE_MAP and (name == "bl" or name == "rm"):
      obj.saveTrace(str("./" + name + "_trace.png").encode('utf8'))

  scan(bl, "bl")
  scan(rm, "rm")
  scan(cddt, "cddt")
  scan(glt, "glt")

# For validation
def random_scan(num_ranges):
  for x in range(10):
    ranges_np = np.zeros(num_ranges, dtype=np.float32)
    queries = np.random.random((num_ranges, 3)).astype(np.float32)
    queries[:,0] *= (testMap.width() - 2.0)
    queries[:,1] *= (testMap.height() - 2.0)
    queries[:,0] += 1.0
    queries[:,1] += 1.0
    queries[:,2] *= 2.0 * np.pi

    def scan(obj, name):
      print("Running " + name + ":")
      print("--------------------")

      start = time.process_time()
      obj.calc_range_many(queries, ranges_np)
      end = time.process_time()
      dur_np = end - start
      print("Numpy: Computed", num_ranges, "ranges in", dur_np, "sec")

      start = time.process_time()
      ranges_slow = np.array([obj.calc_range(*q) for q in queries],
                                dtype=np.float32)
      end = time.process_time()
      dur_slow = end - start
      print("Slow: Computed", num_ranges, "ranges in", dur_slow, "sec")
      print("Numpy speedup:", dur_slow / dur_np)

      diff = np.linalg.norm(ranges_np - ranges_slow)
      print("Diff: {0}".format(diff))
      if diff > 0.001:
        print("Warning: Numpy result different from slow result,",
              "investigation possibly required")
      else:
          print("Test passed")
      print("")

    scan(bl, "bl")
    scan(rm, "rm")
    scan(cddt, "cddt")
    scan(glt, "glt")

fixed_scan(num_ranges=9)
# random_scan(num_ranges=100000)
