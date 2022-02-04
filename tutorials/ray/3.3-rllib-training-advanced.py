import ray


import ray

@ray.remote
class Counter:
    def __init__(self) -> None:
        self.count = 0

    def inc(self, n):
        self.count += n

    def get(self):
        return self.count

counter = Counter.options(name="global_counter").remote()
print(f"Latest count: {ray.get(counter.get.remote())}")
counter = ray.get_actor("global_counter")
counter.inc.remote(3)
print(f"New counter: {ray.get(counter.get.remote())}")