package analysis

// poolAdjacentViolators fits an isotonic (monotone non-decreasing) least
// squares regression to the input sequence using the classical PAV
// algorithm. The input is treated as already sorted by its independent
// axis; the returned slice has the same length and contains the fitted
// values aligned with the input.
//
// Implementation: a stack of (sum, weight, start, end) blocks. Each
// new point is pushed as a singleton block; if its average violates
// the monotone constraint relative to the top of the stack, the two
// blocks are merged and the check repeats. Each point is pushed and
// popped at most once, giving amortized O(n) time.
//
// Used by the calibration recalibration in CalibrateAt: the holdout
// labels are 0/1 binaries sorted by Cox-predicted probability, and
// PAV-fitting them gives a monotone reliability map that
// CalibratedModel applies to fresh predictions.
func poolAdjacentViolators(y []float64) []float64 {
	n := len(y)
	if n == 0 {
		return nil
	}
	type block struct {
		sum    float64
		weight float64
		start  int // first index covered (inclusive)
		end    int // last index covered (exclusive)
	}
	stack := make([]block, 0, n)
	for i, v := range y {
		b := block{sum: v, weight: 1, start: i, end: i + 1}
		// Merge with the top while the previous block's average
		// exceeds the new block's average (monotone violation).
		for len(stack) > 0 {
			top := stack[len(stack)-1]
			if top.sum/top.weight <= b.sum/b.weight {
				break
			}
			b.sum += top.sum
			b.weight += top.weight
			b.start = top.start
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, b)
	}
	fitted := make([]float64, n)
	for _, b := range stack {
		v := b.sum / b.weight
		for k := b.start; k < b.end; k++ {
			fitted[k] = v
		}
	}
	return fitted
}
