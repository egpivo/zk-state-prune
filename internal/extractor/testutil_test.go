package extractor

import "math/rand/v2"

func newTestRand() *rand.Rand {
	return rand.New(rand.NewPCG(1, 2))
}
