import numpy as np



from src.init_av import kmeanspp_init

from src.kmeans_dp import dp_kmeans



                                            

                                                         

                                                                                  

                                                                           

                                                                              

                                                                 





def _project_schedule(raw: np.ndarray, eps_tot: float, eps_min: float, eps_max: float) -> np.ndarray:

    sched = np.clip(raw.astype(float), eps_min, eps_max)

    total = float(np.sum(sched))

    if total <= 0:

        return np.full_like(sched, eps_tot / max(len(sched), 1), dtype=float)

    sched = sched / total * eps_tot

                                                 

    diff = eps_tot - float(np.sum(sched))

    sched[-1] += diff

    return sched





def _evaluate_schedule(

    Z: np.ndarray,

    k: int,

    eps_tot: float,

    T: int,

    schedule: np.ndarray,

    clip_B: float,

    seed: int,

) -> tuple[float, dict]:

    rng = np.random.default_rng(seed)

    init = kmeanspp_init(Z, k, rng)

    result = dp_kmeans(

        Z,

        init_centroids=init,

        k=k,

        T=T,

        eps_iter=eps_tot,

        clip_B=clip_B,

        rng=rng,

        eps_schedule=schedule,

        budget_mode="static",

        proxy_points=Z,

        collapse_boost=1.0,

    )

    labels = result["labels"]

    centroids = result["centroids"]

    counts = np.bincount(labels, minlength=k).astype(float)

    non_empty = int(np.count_nonzero(counts))

    sse = float(np.sum((Z - centroids[labels]) ** 2))

    penalty = 1e6 * max(0, k - non_empty)

    fitness = sse + penalty

    return fitness, {

        "labels": labels,

        "centroids": centroids,

        "non_empty": non_empty,

        "sse": sse,

    }





def optimize_schedule(

    Z: np.ndarray,

    k: int,

    eps_tot: float,

    T: int,

    seed_eval: int = 0,

    eps_min: float = 0.01,

    eps_max: float | None = None,

    pop_size: int = 30,

    generations: int = 40,

    crossover_rate: float = 0.8,

    mutation_rate: float = 0.2,

    elitism: int = 2,

    clip_B: float = 3.0,

    max_ga_samples: int | None = 3000,

) -> dict:

    """
    Genetic algorithm search for eps_schedule that minimizes SSE.
    Returns dict with best schedule and metadata.
    """

    rng = np.random.default_rng(seed_eval)

    eps_max_val = eps_tot if eps_max is None else eps_max

    if eps_max_val <= 0:

        eps_max_val = eps_tot



                                             

    if max_ga_samples is not None and Z.shape[0] > max_ga_samples:

        idx = rng.choice(Z.shape[0], size=max_ga_samples, replace=False)

        Z_eval = Z[idx]

    else:

        Z_eval = Z



    def init_population() -> np.ndarray:

        pop = []

        for _ in range(pop_size):

            raw = rng.dirichlet(np.ones(T)) * eps_tot

            sched = _project_schedule(raw, eps_tot, eps_min, eps_max_val)

            pop.append(sched)

        return np.stack(pop, axis=0)



    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        if rng.random() > crossover_rate:

            return parent1.copy(), parent2.copy()

        point = rng.integers(1, T)

        child1 = np.concatenate([parent1[:point], parent2[point:]])

        child2 = np.concatenate([parent2[:point], parent1[point:]])

        return child1, child2



    def mutate(child: np.ndarray) -> np.ndarray:

        if rng.random() > mutation_rate:

            return child

        noise = rng.normal(loc=0.0, scale=eps_tot / max(T, 1) * 0.1, size=T)

        mutated = child + noise

        return _project_schedule(mutated, eps_tot, eps_min, eps_max_val)



    population = init_population()

    best_sched = population[0]

    best_fitness = np.inf



    for _ in range(generations):

        fitness_list = []

        for sched in population:

            fit, _ = _evaluate_schedule(

                Z_eval,

                k,

                eps_tot,

                T,

                schedule=sched,

                clip_B=clip_B,

                seed=seed_eval,

            )

            fitness_list.append(fit)

        fitness_arr = np.array(fitness_list)

        order = np.argsort(fitness_arr)

        population = population[order]

        fitness_arr = fitness_arr[order]



        if fitness_arr[0] < best_fitness:

            best_fitness = float(fitness_arr[0])

            best_sched = population[0].copy()



                 

        new_pop = [population[i].copy() for i in range(min(elitism, pop_size))]



                              

        while len(new_pop) < pop_size:

            i, j = rng.integers(0, pop_size, size=2)

            parent1 = population[i]

            parent2 = population[j]

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1)

            child2 = mutate(child2)

            new_pop.append(child1)

            if len(new_pop) < pop_size:

                new_pop.append(child2)

        population = np.stack(new_pop, axis=0)



                                                      

    final_fit, final_info = _evaluate_schedule(

        Z,

        k,

        eps_tot,

        T,

        schedule=best_sched,

        clip_B=clip_B,

        seed=seed_eval,

    )

    return {

        "schedule": _project_schedule(best_sched, eps_tot, eps_min, eps_max_val),

        "fitness": final_fit,

        "best_info": final_info,

    }

