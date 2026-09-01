<?php

namespace Tests\Feature;

use App\Models\User;
use App\Models\Violation;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class ViolationReviewTest extends TestCase
{
    use RefreshDatabase;

    private function officer(): User
    {
        return User::factory()->create(['role' => 'officer']);
    }

    public function test_guest_cannot_view_evidence_detail(): void
    {
        $violation = Violation::factory()->create();

        $this->get(route('violations.show', $violation))->assertRedirect('/login');
    }

    public function test_officer_can_view_complete_evidence_detail(): void
    {
        $violation = Violation::factory()->create([
            'event_id' => 'DETAIL-EVENT-001',
            'plate_number' => 'DHAKA-GA-1234',
            'ocr_raw_text' => 'ঢাকা গ ১২৩৪',
        ]);

        $this->actingAs($this->officer())
            ->get(route('violations.show', $violation))
            ->assertOk()
            ->assertSee('DETAIL-EVENT-001')
            ->assertSee('DHAKA-GA-1234')
            ->assertSee('Officer review');
    }

    public function test_officer_can_correct_plate_and_confirm_violation(): void
    {
        $officer = $this->officer();
        $violation = Violation::factory()->create([
            'plate_number' => 'UNREADABLE',
            'status' => 'PENDING',
        ]);

        $response = $this->actingAs($officer)->patch(
            route('violations.update', $violation),
            [
                'plate_number' => 'ঢাকা মেট্রো গ 12-3456',
                'status' => 'CONFIRMED',
                'officer_notes' => 'Plate manually verified from evidence.',
            ],
        );

        $response->assertRedirect(route('violations.show', $violation));
        $this->assertDatabaseHas('violations', [
            'id' => $violation->id,
            'plate_number' => 'ঢাকা মেট্রো গ 12-3456',
            'original_plate_number' => 'UNREADABLE',
            'status' => 'CONFIRMED',
            'reviewed_by' => $officer->id,
            'officer_notes' => 'Plate manually verified from evidence.',
        ]);
        $this->assertNotNull($violation->fresh()->reviewed_at);
        $this->assertNotNull($violation->fresh()->plate_corrected_at);
    }

    public function test_officer_can_dismiss_violation_without_changing_plate(): void
    {
        $officer = $this->officer();
        $violation = Violation::factory()->create(['plate_number' => 'DHAKA-1234']);

        $this->actingAs($officer)->patch(
            route('violations.update', $violation),
            [
                'plate_number' => 'DHAKA-1234',
                'status' => 'DISMISSED',
                'officer_notes' => 'Signal state was manually found incorrect.',
            ],
        )->assertSessionHasNoErrors();

        $this->assertDatabaseHas('violations', [
            'id' => $violation->id,
            'status' => 'DISMISSED',
            'reviewed_by' => $officer->id,
        ]);
        $this->assertNull($violation->fresh()->original_plate_number);
    }

    public function test_review_rejects_invalid_status(): void
    {
        $violation = Violation::factory()->create();

        $this->actingAs($this->officer())->patch(
            route('violations.update', $violation),
            [
                'plate_number' => 'DHAKA-1234',
                'status' => 'DELETED',
            ],
        )->assertSessionHasErrors('status');
    }
}
