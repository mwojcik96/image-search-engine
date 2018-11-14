class Image:
    def __init__(self, filename):
        self.filename = filename
        self.features = dict()

    def fill_measure(self, name, vector):
        self.features[name] = vector