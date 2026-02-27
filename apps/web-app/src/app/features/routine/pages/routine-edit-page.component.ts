import { Component, OnInit, computed, inject, signal } from "@angular/core";
import { RouterLink } from "@angular/router";
import { UiButtonComponent } from "../../../shared/ui/button/button.component";
import { StatePanelComponent } from "../../../shared/components/state-panel.component";
import { UiSkeletonComponent } from "../../../shared/ui/skeleton/skeleton.component";
import { RoutineStore } from "../data/routine.store";

type EditableRoutineStep = {
  id?: string;
  title: string;
  durationMinutes: number;
  emoji: string;
};

@Component({
  selector: "app-routine-edit-page",
  standalone: true,
  imports: [
    RouterLink,
    UiButtonComponent,
    StatePanelComponent,
    UiSkeletonComponent,
  ],
  template: `
    <section class="screen screen--dark sleep-summary-theme">
      @switch (viewState()) {
        @case ("loading") {
          <div class="screen__section">
            <ui-skeleton [height]="24" />
            <ui-skeleton [height]="140" />
          </div>
        }
        @case ("error") {
          <app-state-panel
            title="Routine editor unavailable"
            [message]="store.errorMessage() ?? 'Could not load routine editor.'"
            actionLabel="Retry"
            (action)="reload()"
          />
        }
        @default {
          <div class="screen__header">
            <div>
              <p class="screen__sub">Edit Routine</p>
              <h1 class="screen__title">{{ routineTitle() }}</h1>
            </div>
            <a class="primary-link" routerLink="/routine">Back</a>
          </div>

          <form class="screen__section" (submit)="onSubmit($event)">
            <label class="form-field">
              <span class="form-field__label">Routine title</span>
              <input
                class="form-field__input"
                type="text"
                [value]="titleInput()"
                maxlength="80"
                required
                (input)="onTitleInput($event)"
              />
            </label>

            <div class="chart-card">
              <div class="section-header">
                <h3>Steps</h3>
                <button class="chip" type="button" (click)="addStep()">
                  Add step
                </button>
              </div>

              @for (step of editableSteps(); track $index) {
                <div class="routine-step-edit">
                  <label class="form-field routine-step-edit__title">
                    <span class="form-field__label">Step title</span>
                    <input
                      class="form-field__input"
                      type="text"
                      [value]="step.title"
                      maxlength="60"
                      required
                      (input)="onStepTitleInput($index, $event)"
                    />
                  </label>

                  <label class="form-field routine-step-edit__duration">
                    <span class="form-field__label">Minutes</span>
                    <input
                      class="form-field__input"
                      type="number"
                      min="1"
                      max="60"
                      [value]="step.durationMinutes"
                      required
                      (input)="onStepDurationInput($index, $event)"
                    />
                  </label>

                  <label class="form-field routine-step-edit__emoji">
                    <span class="form-field__label">Emoji</span>
                    <input
                      class="form-field__input"
                      type="text"
                      maxlength="3"
                      [value]="step.emoji"
                      (input)="onStepEmojiInput($index, $event)"
                    />
                  </label>

                  <button
                    class="chip chip--danger"
                    type="button"
                    [disabled]="editableSteps().length <= 1"
                    (click)="removeStep($index)"
                  >
                    Remove
                  </button>
                </div>
              }
            </div>

            @if (validationMessage()) {
              <p class="form-error" role="alert">{{ validationMessage() }}</p>
            }

            <div class="screen__cta">
              <ui-button [disabled]="!canSubmit() || store.isSaving()">
                {{ store.isSaving() ? "Saving..." : "Save routine" }}
              </ui-button>
              <a class="primary-link" routerLink="/routine">Cancel</a>
            </div>
          </form>
        }
      }
    </section>
  `,
})
export class RoutineEditPageComponent implements OnInit {
  readonly store = inject(RoutineStore);
  readonly viewState = this.store.status;

  readonly titleInput = signal("");
  readonly editableSteps = signal<EditableRoutineStep[]>([]);

  readonly routineTitle = computed(
    () => this.store.routine()?.title ?? "Edit routine",
  );

  readonly validationMessage = computed(() => {
    if (!this.titleInput().trim()) {
      return "Routine title is required.";
    }

    if (this.editableSteps().length === 0) {
      return "Add at least one routine step.";
    }

    for (const step of this.editableSteps()) {
      if (!step.title.trim()) {
        return "Each step needs a title.";
      }
      if (
        Number.isNaN(step.durationMinutes) ||
        step.durationMinutes < 1 ||
        step.durationMinutes > 60
      ) {
        return "Step duration must be between 1 and 60 minutes.";
      }
    }

    return null;
  });

  readonly canSubmit = computed(() => this.validationMessage() === null);

  ngOnInit(): void {
    void this.store.loadRoutine().then(() => this.hydrateFromStore());
  }

  reload(): void {
    void this.store.loadRoutine().then(() => this.hydrateFromStore());
  }

  onTitleInput(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.titleInput.set(target.value);
  }

  onStepTitleInput(index: number, event: Event): void {
    const target = event.target as HTMLInputElement;
    this.updateStep(index, { title: target.value });
  }

  onStepDurationInput(index: number, event: Event): void {
    const target = event.target as HTMLInputElement;
    const duration = Number(target.value);
    this.updateStep(index, {
      durationMinutes: Number.isNaN(duration) ? 0 : duration,
    });
  }

  onStepEmojiInput(index: number, event: Event): void {
    const target = event.target as HTMLInputElement;
    this.updateStep(index, { emoji: target.value.trim() });
  }

  addStep(): void {
    this.editableSteps.update((steps) => [
      ...steps,
      {
        title: "",
        durationMinutes: 5,
        emoji: "",
      },
    ]);
  }

  removeStep(index: number): void {
    this.editableSteps.update((steps) => steps.filter((_, i) => i !== index));
  }

  async onSubmit(event: Event): Promise<void> {
    event.preventDefault();
    if (!this.canSubmit()) {
      return;
    }

    await this.store.updateRoutine({
      title: this.titleInput().trim(),
      steps: this.editableSteps().map((step) => ({
        id: step.id,
        title: step.title.trim(),
        durationMinutes: step.durationMinutes,
        emoji: step.emoji || null,
      })),
    });

    if (this.store.status() === "success") {
      this.hydrateFromStore();
    }
  }

  private hydrateFromStore(): void {
    const routine = this.store.routine();
    if (!routine) {
      return;
    }
    this.titleInput.set(routine.title);
    this.editableSteps.set(
      routine.steps.map((step) => ({
        id: step.id,
        title: step.title,
        durationMinutes: step.duration_minutes,
        emoji: step.emoji ?? "",
      })),
    );
  }

  private updateStep(index: number, patch: Partial<EditableRoutineStep>): void {
    this.editableSteps.update((steps) =>
      steps.map((step, i) => (i === index ? { ...step, ...patch } : step)),
    );
  }
}
