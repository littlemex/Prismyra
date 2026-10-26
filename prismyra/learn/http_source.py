"""A second `prismyra.learn.experience.ExperienceSource`: pulling logged experience over HTTP/JSON instead
of reading local files. This exists to show the read side is genuinely swappable, not tied to
`LocalExperienceSource`'s own on-disk layout -- and to give an operator who already ships experience
somewhere else (an object store behind an API, a pipeline's own query service) a source that does not
require adopting this project's directory convention to use `prismyra.learn.fit`/`.idle` against it.

**The contract, deliberately generic.** A paginated pull against one URL:

    GET {base_url}?task={task}&since={ts}&page={token}
    -> {"rows": [<prismyra.learn.experience.Experience.to_json() shape>, ...], "next_page": <token> | null}

``since`` and ``page`` are both query parameters an implementation on the other end is free to interpret
however it likes (a timestamp cutoff and an opaque cursor are the only properties this reads): what this
class needs from the far side is exactly what `prismyra.learn.experience.LocalExperienceSource` already
produces locally -- a stream of `Experience.to_json()` objects, same schema, same `request_id` field feedback
joins by. No new dependency: ``urllib`` only.

**Why not a CloudWatch Logs or Prometheus-shaped source instead.** Either is a real, useful thing to build
for an operator who already has one of those -- but it is also a *second* dependency (the AWS SDK; a
Prometheus HTTP client) this package would otherwise never need, for a destination most operators of this
project do not have. The generic pull contract above is what both would be built *on top of* (a CloudWatch
reader turns a `since`/`page` pair into a `FilterLogEvents` call with its own `nextToken`; a Prometheus
reader turns the same pair into a range-query's own time window) -- this module ships the contract and the
one implementation against it that needs nothing else installed, and leaves the destination-specific
adapters as exactly that: an adapter satisfying `ExperienceSource`, not a rewrite of the interface.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator

from .experience import Experience


class HttpSourceError(Exception):
    """The far side did not answer the contract this class reads -- unreachable, not JSON, or missing
    ``"rows"``. Raised eagerly, mid-iteration, rather than silently yielding nothing: a caller training on
    "whatever came back" from a source that is actually broken would admit or reject a student on far less
    data than it believes it saw.
    """


class HttpPullExperienceSource:
    """Pulls every row for one ``task`` from ``base_url``, paging until the far side reports no more.
    ``timeout`` bounds one page request; a single slow or hanging page fails the whole pull rather than
    blocking a training run indefinitely.
    """

    def __init__(self, base_url: str, *, timeout: float = 10.0):
        self.base_url = base_url
        self.timeout = timeout

    def _fetch_page(self, task: str, since: float | None, page: str | None) -> dict:
        params: dict[str, str] = {"task": task}
        if since is not None:
            params["since"] = str(since)
        if page is not None:
            params["page"] = page
        url = self.base_url + ("&" if "?" in self.base_url else "?") + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as e:
            raise HttpSourceError(f"{url}: {e}") from e
        if "rows" not in raw:
            raise HttpSourceError(f"{url}: response has no 'rows' field: {raw!r}")
        return raw

    def iter_experiences(self, task: str, *, since: float | None = None) -> Iterator[Experience]:
        page: str | None = None
        while True:
            raw = self._fetch_page(task, since, page)
            for row in raw["rows"]:
                yield Experience.from_json(row)
            page = raw.get("next_page")
            if page is None:
                return
