"""Validate declared plugin and distribution requirements without importing packages."""

from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


def plugin_requirements(info, allow_missing_optional):
    result = []
    for optional, entries in ((False, info.dependencies), (True, info.optional_dependencies)):
        for value in entries:
            requirement = Requirement(value)
            if requirement.url or requirement.extras:
                raise ValueError("Plugin dependencies do not accept URLs or distribution extras")
            if requirement.marker is None or requirement.marker.evaluate():
                result.append((requirement, optional and allow_missing_optional))
    return result


def distribution_satisfied(value, strict):
    requirement = Requirement(value)
    if requirement.url:
        raise ValueError("Python dependencies must use installed distribution requirements")
    if requirement.marker is not None and not requirement.marker.evaluate():
        return True
    try:
        distribution = metadata.distribution(requirement.name)
    except metadata.PackageNotFoundError:
        return False
    if strict and not requirement.specifier.contains(
        Version(distribution.version), prereleases=True
    ):
        return False
    extras = {
        canonicalize_name(extra) for extra in distribution.metadata.get_all("Provides-Extra") or []
    }
    if not {canonicalize_name(extra) for extra in requirement.extras} <= extras:
        return False
    for extra in requirement.extras:
        for dependency in distribution.requires or []:
            selected = Requirement(dependency)
            if selected.marker is not None and not selected.marker.evaluate({"extra": extra}):
                continue
            if selected.url:
                return False
            try:
                installed = metadata.version(selected.name)
            except metadata.PackageNotFoundError:
                return False
            if strict and not selected.specifier.contains(Version(installed), prereleases=True):
                return False
    return True
