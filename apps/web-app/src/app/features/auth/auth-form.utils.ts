export const MIN_PASSWORD_LENGTH = 8;
export const DEFAULT_AUTH_REDIRECT = "/dashboard";

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export const isEmailValid = (email: string): boolean => EMAIL_PATTERN.test(email.trim());

export const isPasswordValid = (password: string): boolean =>
  password.trim().length >= MIN_PASSWORD_LENGTH;

export const doPasswordsMatch = (password: string, confirmPassword: string): boolean =>
  password === confirmPassword;

export const getPostAuthRedirect = (candidate?: string | null): string => {
  if (!candidate || typeof candidate !== "string") {
    return DEFAULT_AUTH_REDIRECT;
  }

  if (!candidate.startsWith("/")) {
    return DEFAULT_AUTH_REDIRECT;
  }

  if (candidate.startsWith("//")) {
    return DEFAULT_AUTH_REDIRECT;
  }

  if (candidate.startsWith("/login") || candidate.startsWith("/signup")) {
    return DEFAULT_AUTH_REDIRECT;
  }

  return candidate;
};
