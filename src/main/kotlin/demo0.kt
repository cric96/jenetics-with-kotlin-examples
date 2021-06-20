import io.jenetics.BitChromosome
import io.jenetics.BitGene
import io.jenetics.Genotype
import io.jenetics.engine.Engine
import io.jenetics.engine.EvolutionResult
import io.jenetics.engine.EvolutionStatistics

//Demo O: Bit count problem, taken from https://jenetics.io/manual/manual-6.2.0.pdf page 2

fun main(args: Array<String>) {
    //constants
    val chromosomeSize = 10
    val trueProbability = 0.5
    val upperBoundExecutionLimit = 100L
    //fitness function, it should take the solution (genotype) as input an return a pay-off (fitness) of how much the solution is near to a good solution
    fun fitness(solution : Genotype<BitGene>) : Int = solution
        .chromosome().`as`(BitChromosome::class.java)
        .bitCount()
    //factory for creating solution
    val factory = Genotype.of(BitChromosome.of(chromosomeSize, trueProbability))
    //create the execution context
    val engine = Engine.builder(::fitness, factory)
        .build()
    //create the execution stream
    /*
    GA pseudo algorithm:
    Po <- initialPopulation
    F(Po) //calculate fitness
    while !finished
        g <- g + 1 //change generation
        Sg <- select(Pg - 1) //survivor
        Og <- select(Pg - 1) //offspring
        Og <- alter(Og) //mutation + crossover
        Pg <- filter[gi > gmax](Sg) + filter[gi > gmax](Og) // combine the survivor with the alter population
        F(Pg)
     */
    val statistics = EvolutionStatistics.ofNumber<Int>()
    val result = engine
        .stream()
        .limit(upperBoundExecutionLimit)
        .peek(statistics)
        .collect(EvolutionResult.toBestEvolutionResult()) //it can be improved with pimping

    println(statistics) //unsafe..
    println("Result = $result")
}