from runtime_loader import runtime_loader


def run(args):
    loader = create_loader(args)
    if loader is None:
        return None

    return loader.run()


def create_loader(args):
    return runtime_loader.RuntimeLoader(args)
