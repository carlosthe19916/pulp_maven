"""Tests for the repair_packages repository action (PULP-2478 / pulp_maven#508).

These tests create the stranded state on current code with the "build then break" approach:
upload POMs normally (their MavenPackages auto-associate), then remove the MavenPackage content
units via ``modify`` while keeping the POM artifacts. The MavenPackage rows still exist globally but
are no longer associated with the version -- exactly the production state the 0011 migration created.

Tests are ordered to follow the sequence of outcomes in ``repair_packages``: first the no-op branch
(nothing to repair), then the repair branch (associate missing packages; POM-less GAVs ignored), then
a repair followed by a no-op on re-run (idempotency).
"""

import uuid

import pytest


def _uid():
    return uuid.uuid4().hex[:8]


@pytest.fixture
def stranded_repo(
    maven_repo_factory,
    maven_artifact_api_client,
    maven_package_api_client,
    maven_repo_api_client,
    pom_file_factory,
    monitor_task,
):
    """Create a repo whose latest version has POMs but no associated MavenPackages.

    Args:
        gavs: list of (group_id, artifact_id, version) tuples.

    Returns:
        the repository, latest version containing POMs but zero MavenPackages.
    """

    def _factory(gavs):
        repo = maven_repo_factory()
        uid = _uid()

        content_hrefs = []
        for group_id, artifact_id, version in gavs:
            full_group = f"{group_id}.{uid}"  # unique per run for @pytest.mark.parallel safety
            pom_path = pom_file_factory(
                group_id=full_group,
                artifact_id=artifact_id,
                version=version,
                name=f"{artifact_id} {version}",
                packaging="jar",
            )
            group_path = full_group.replace(".", "/")
            relative_path = f"{group_path}/{artifact_id}/{version}/{artifact_id}-{version}.pom"
            content = maven_artifact_api_client.upload(
                file=str(pom_path), relative_path=relative_path
            )
            content_hrefs.append(content.pulp_href)

        monitor_task(
            maven_repo_api_client.modify(repo.pulp_href, {"add_content_units": content_hrefs}).task
        )

        # Packages were auto-associated; strand them by removing the MavenPackage units.
        repo = maven_repo_api_client.read(repo.pulp_href)
        package_hrefs = [
            p.pulp_href
            for p in maven_package_api_client.list(
                repository_version=repo.latest_version_href
            ).results
        ]
        assert len(package_hrefs) == len(gavs)
        monitor_task(
            maven_repo_api_client.modify(
                repo.pulp_href, {"remove_content_units": package_hrefs}
            ).task
        )

        repo = maven_repo_api_client.read(repo.pulp_href)
        assert (
            maven_package_api_client.list(repository_version=repo.latest_version_href).count == 0
        ), "repo should be stranded (POMs present, no MavenPackages) before repair"
        return repo

    return _factory


@pytest.mark.parallel
def test_repair_packages_empty_repo_noop(
    maven_repo_factory,
    maven_package_api_client,
    maven_repo_api_client,
    monitor_task,
):
    """repair_packages on an empty repository succeeds and changes nothing."""
    repo = maven_repo_factory()
    version_before = repo.latest_version_href

    monitor_task(maven_repo_api_client.repair_packages(repo.pulp_href).task)

    repo = maven_repo_api_client.read(repo.pulp_href)
    assert repo.latest_version_href == version_before, "empty repo should get no new version"
    assert maven_package_api_client.list(repository_version=repo.latest_version_href).count == 0


@pytest.mark.parallel
def test_repair_packages_associates_missing_packages(
    stranded_repo,
    maven_package_api_client,
    maven_repo_api_client,
    monitor_task,
):
    """repair_packages associates the MavenPackages for stranded POMs in a new version."""
    repo = stranded_repo([("com.example", "alpha", "1.0.0"), ("com.example", "beta", "2.0.0")])
    version_before = repo.latest_version_href

    monitor_task(maven_repo_api_client.repair_packages(repo.pulp_href).task)

    repo = maven_repo_api_client.read(repo.pulp_href)
    assert repo.latest_version_href != version_before, "repair should create a new version"
    packages = maven_package_api_client.list(repository_version=repo.latest_version_href)
    assert packages.count == 2, "both stranded packages should now be associated"


@pytest.mark.parallel
def test_repair_packages_ignores_pomless_gav(
    maven_repo_factory,
    maven_artifact_api_client,
    maven_package_api_client,
    maven_repo_api_client,
    pom_file_factory,
    monitor_task,
    tmp_path,
):
    """repair_packages only backfills POM-backed GAVs; a .jar-only GAV gets no package."""
    repo = maven_repo_factory()
    uid = _uid()
    group = f"com.example.{uid}"
    group_path = group.replace(".", "/")

    # A POM-backed GAV (will be stranded) ...
    pom_path = pom_file_factory(
        group_id=group, artifact_id="has-pom", version="1.0.0", name="Has Pom", packaging="jar"
    )
    pom_content = maven_artifact_api_client.upload(
        file=str(pom_path),
        relative_path=f"{group_path}/has-pom/1.0.0/has-pom-1.0.0.pom",
    )
    # ... and a GAV with only a .jar (no .pom) -> never gets a package.
    jar_path = tmp_path / "no-pom-1.0.0.jar"
    jar_path.write_bytes(b"dummy jar bytes, not a real archive")
    jar_content = maven_artifact_api_client.upload(
        file=str(jar_path),
        relative_path=f"{group_path}/no-pom/1.0.0/no-pom-1.0.0.jar",
    )

    monitor_task(
        maven_repo_api_client.modify(
            repo.pulp_href, {"add_content_units": [pom_content.pulp_href, jar_content.pulp_href]}
        ).task
    )

    # Strand the single auto-created package (for has-pom); no-pom never had one.
    repo = maven_repo_api_client.read(repo.pulp_href)
    package_hrefs = [
        p.pulp_href
        for p in maven_package_api_client.list(repository_version=repo.latest_version_href).results
    ]
    assert len(package_hrefs) == 1
    monitor_task(
        maven_repo_api_client.modify(repo.pulp_href, {"remove_content_units": package_hrefs}).task
    )
    repo = maven_repo_api_client.read(repo.pulp_href)
    assert maven_package_api_client.list(repository_version=repo.latest_version_href).count == 0

    monitor_task(maven_repo_api_client.repair_packages(repo.pulp_href).task)

    repo = maven_repo_api_client.read(repo.pulp_href)
    packages = maven_package_api_client.list(repository_version=repo.latest_version_href)
    assert packages.count == 1, "only the POM-backed GAV should get a package"
    assert packages.results[0].artifact_id == "has-pom", "the .jar-only GAV must not get a package"


@pytest.mark.parallel
def test_repair_packages_is_idempotent(
    stranded_repo,
    maven_package_api_client,
    maven_repo_api_client,
    monitor_task,
):
    """A second repair_packages run finds nothing stranded and creates no new version."""
    repo = stranded_repo([("com.example", "gamma", "1.0.0")])

    monitor_task(maven_repo_api_client.repair_packages(repo.pulp_href).task)
    repo = maven_repo_api_client.read(repo.pulp_href)
    version_after_repair = repo.latest_version_href
    assert maven_package_api_client.list(repository_version=version_after_repair).count == 1

    monitor_task(maven_repo_api_client.repair_packages(repo.pulp_href).task)
    repo = maven_repo_api_client.read(repo.pulp_href)
    assert repo.latest_version_href == version_after_repair, "second run must not create a version"
    assert maven_package_api_client.list(repository_version=repo.latest_version_href).count == 1
