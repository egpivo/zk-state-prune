// Package config is the Phase-2 YAML loader for zksp. It is intentionally
// narrow: only the knobs the statistical policy actually reads are
// unmarshaled, because every other CLI value already has a reasonable
// hardcoded default. Grow this file when Phase 3 needs to expose
// additional YAML sections to the CLI.
package config

import (
	"errors"
	"fmt"
	"io/fs"
	"os"

	"gopkg.in/yaml.v3"
)

// Config mirrors the subset of configs/default.yaml we actually
// consume. Sections that the CLI flags already handle (observation
// window, extractor mock parameters, survival hyperparameters) stay
// out of this struct until a caller demonstrates they need them.
type Config struct {
	Logging struct {
		Level string `yaml:"level"`
	} `yaml:"logging"`

	Pruning struct {
		Cost struct {
			// RAMUnitCost is the cost of keeping one slot in the hot
			// tier for one block.
			RAMUnitCost float64 `yaml:"ram_unit_cost"`
			// MissPenalty is the cost of a single cold-tier fetch.
			MissPenalty float64 `yaml:"miss_penalty"`
		} `yaml:"cost"`
	} `yaml:"pruning"`
}

// Default returns the canonical built-in configuration. Every CLI
// invocation starts from this baseline and then overlays values from a
// YAML file (if --config was passed) and finally from explicit flags.
func Default() *Config {
	c := &Config{}
	c.Logging.Level = "info"
	c.Pruning.Cost.RAMUnitCost = 1.0
	c.Pruning.Cost.MissPenalty = 1e5
	return c
}

// Load reads a YAML config from path and merges it into a copy of
// Default(). Unspecified fields keep their default values. Missing
// files are a soft error: callers check for os.ErrNotExist to decide
// whether to fall back to defaults silently (the main command does
// this when --config is left unset but a default file isn't present).
func Load(path string) (*Config, error) {
	c := Default()
	if path == "" {
		return c, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return c, err
		}
		return nil, fmt.Errorf("config: read %s: %w", path, err)
	}
	if err := yaml.Unmarshal(b, c); err != nil {
		return nil, fmt.Errorf("config: parse %s: %w", path, err)
	}
	return c, nil
}
