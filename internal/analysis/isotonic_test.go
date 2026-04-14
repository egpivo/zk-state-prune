package analysis

import (
	"math"
	"testing"
)

func TestPAV_AlreadyMonotoneIsIdentity(t *testing.T) {
	in := []float64{0.1, 0.2, 0.5, 0.7, 1.0}
	got := poolAdjacentViolators(in)
	for i := range in {
		if math.Abs(got[i]-in[i]) > 1e-12 {
			t.Errorf("PAV[%d]=%v, want %v", i, got[i], in[i])
		}
	}
}

func TestPAV_AntiMonotoneCollapsesToMean(t *testing.T) {
	in := []float64{1.0, 0.8, 0.6, 0.4, 0.2}
	got := poolAdjacentViolators(in)
	want := 0.6 // mean
	for i, v := range got {
		if math.Abs(v-want) > 1e-12 {
			t.Errorf("PAV[%d]=%v, want %v", i, v, want)
		}
	}
}

func TestPAV_CanonicalExample(t *testing.T) {
	// Standard PAV example. [4,5,1,6,7] → [10/3, 10/3, 10/3, 6, 7].
	got := poolAdjacentViolators([]float64{4, 5, 1, 6, 7})
	want := []float64{10.0 / 3, 10.0 / 3, 10.0 / 3, 6, 7}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Errorf("PAV[%d]=%v, want %v", i, got[i], want[i])
		}
	}
	// Output must be non-decreasing.
	for i := 1; i < len(got); i++ {
		if got[i] < got[i-1]-1e-12 {
			t.Errorf("PAV output not monotone: %v", got)
			break
		}
	}
}

func TestPAV_PreservesSumWeighted(t *testing.T) {
	// PAV is a least-squares projection: it must preserve the total sum
	// (since each merged block keeps the same sum, just averaged).
	in := []float64{0.9, 0.1, 0.5, 0.3, 0.8, 0.2, 0.7}
	got := poolAdjacentViolators(in)
	var inSum, outSum float64
	for i := range in {
		inSum += in[i]
		outSum += got[i]
	}
	if math.Abs(inSum-outSum) > 1e-9 {
		t.Errorf("PAV did not preserve sum: in=%v out=%v", inSum, outSum)
	}
	// And the result must be non-decreasing.
	for i := 1; i < len(got); i++ {
		if got[i] < got[i-1]-1e-12 {
			t.Errorf("PAV not monotone: %v", got)
			break
		}
	}
}

func TestPAV_EmptyInput(t *testing.T) {
	if got := poolAdjacentViolators(nil); got != nil {
		t.Errorf("PAV(nil) = %v, want nil", got)
	}
}

func TestPAV_SingletonIdentity(t *testing.T) {
	got := poolAdjacentViolators([]float64{0.42})
	if len(got) != 1 || got[0] != 0.42 {
		t.Errorf("PAV([0.42]) = %v", got)
	}
}
