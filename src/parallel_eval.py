def evaluate_genome_worker(args):
    evaluator, genome = args
    return evaluator.evaluate(genome)
