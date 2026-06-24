"""Allow `python -m pylanetstack` to behave like the CLI executable."""

from .cli import main


raise SystemExit(main())
