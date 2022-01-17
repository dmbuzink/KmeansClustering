from pyspark import SparkConf, SparkContext, RDD
from typing import Iterable, List

sparkConf = SparkConf().setAppName('test')
spark = SparkContext(conf=sparkConf)

def concat_lists(t: Iterable[List[int]]) -> Iterable[List[int]]:
    return [[item for sublist in t for item in sublist]]

a = [i for i in range(20)]
num_partitions = 16

b = spark.parallelize(a, num_partitions).glom()

while num_partitions > 1:
    num_partitions = max(1, num_partitions // 2)
    b = b.repartition(num_partitions).mapPartitions(concat_lists)

c = b.collect()

print(c)

spark.stop()