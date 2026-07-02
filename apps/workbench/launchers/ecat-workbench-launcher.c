#include <mach-o/dyld.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int executable_dir(char *out, size_t out_size) {
    uint32_t size = 0;
    _NSGetExecutablePath(NULL, &size);
    char *raw_path = (char *)calloc((size_t)size + 1, sizeof(char));
    if (raw_path == NULL) {
        return -1;
    }
    if (_NSGetExecutablePath(raw_path, &size) != 0) {
        free(raw_path);
        return -1;
    }

    char resolved[PATH_MAX];
    const char *source = realpath(raw_path, resolved) == NULL ? raw_path : resolved;
    if (strlen(source) + 1 > out_size) {
        free(raw_path);
        return -1;
    }
    strcpy(out, source);
    free(raw_path);

    char *last_slash = strrchr(out, '/');
    if (last_slash == NULL) {
        return -1;
    }
    *last_slash = '\0';
    return 0;
}

int main(void) {
    char macos_dir[PATH_MAX];
    char script_path[PATH_MAX];

    if (executable_dir(macos_dir, sizeof(macos_dir)) != 0) {
        fprintf(stderr, "Could not resolve eCAT app bundle executable path.\n");
        return 1;
    }
    if (snprintf(
            script_path,
            sizeof(script_path),
            "%s/../../../ecat-workbench-launcher.sh",
            macos_dir
        ) >= (int)sizeof(script_path)) {
        fprintf(stderr, "Resolved eCAT launcher path is too long.\n");
        return 1;
    }

    setenv("ECAT_LAUNCHER_ENTRY", "eCAT app bundle executable started.", 1);
    execl("/bin/zsh", "zsh", script_path, (char *)NULL);
    perror("Could not execute eCAT launcher script");
    return 1;
}
