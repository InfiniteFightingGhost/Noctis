let analyticsInitialized = false;

export async function bootstrapAnalytics() {
  if (analyticsInitialized) {
    return;
  }

  analyticsInitialized = true;
}
