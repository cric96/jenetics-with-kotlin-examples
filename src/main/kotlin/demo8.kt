import io.jenetics.Mutator
import io.jenetics.engine.Engine
import io.jenetics.engine.EvolutionResult
import io.jenetics.engine.Limits
import io.jenetics.ext.SingleNodeCrossover
import io.jenetics.prog.op.*
import io.jenetics.prog.regression.Error
import io.jenetics.prog.regression.LossFunction
import io.jenetics.prog.regression.Regression
import io.jenetics.prog.regression.Sample
import io.jenetics.util.ISeq
import io.jenetics.util.RandomRegistry
import kotlin.math.pow

/** example taken from https://jenetics.io/manual/manual-6.2.0.pdf page 124, symbolic regression*/
//improved version of demo1
const val bound = 10
const val depth = 5
const val rangeSize = 10
val dataSet : List<Sample<Double>> = ((-rangeSize) to (rangeSize)).toList().map { it / rangeSize.toDouble() }.map { x -> Sample.ofDouble(x, 4 * x.pow(3) - 3 * x.pow(2) + x) }
val Ops : ISeq<Op<Double>> = ISeq.of(MathOp.ADD, MathOp.SUB, MathOp.MUL) //operation supported
val Terminals : ISeq<Op<Double>> = ISeq.of(Var.of("x", 0), EphemeralConst.of { RandomRegistry.random().nextInt(bound).toDouble()})
val RegressionProblem: Regression<Double> = Regression.of(
    Regression.codecOf(
        Ops, Terminals, depth
    ) { t -> t.gene().size() < 30 },
    Error.of { calculated, expected -> LossFunction.mse(calculated, expected) },
    dataSet
)
fun main() {
    val crossoverProbability = 0.1
    val threshold = 0.1
    val engine = Engine.builder(RegressionProblem)
        .minimizing()
        .alterers(
            SingleNodeCrossover(crossoverProbability),
            Mutator()
        )
        .build()

    val result = engine.stream()
        .limit(Limits.byFitnessThreshold(threshold))
        .collect(EvolutionResult.toBestEvolutionResult())

    val program = result.bestPhenotype()
        .genotype()
        .gene()

    val tree = program.toTreeNode()
    MathExpr.rewrite(tree)
    println("G: ${result.totalGenerations()}")
    println("F: ${MathExpr(tree)}")
}