<?php

namespace Tests\Feature;

use App\Models\User;
use App\Models\Violation;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class ViolationIndexTest extends TestCase
{
    use RefreshDatabase;

    private function officer(): User
    {
        return User::factory()->create(['role' => 'officer']);
    }

    public function test_guest_cannot_open_violation_register(): void
    {
        $this->get('/violations')->assertRedirect('/login');
    }

    public function test_officer_sees_violation_records_newest_first(): void
    {
        Violation::factory()->create([
            'plate_number' => 'OLD-1000',
            'frame_timestamp' => now()->subDay(),
        ]);
        Violation::factory()->create([
            'plate_number' => 'NEW-2000',
            'frame_timestamp' => now(),
        ]);

        $this->actingAs($this->officer())
            ->get('/violations')
            ->assertOk()
            ->assertSeeInOrder(['NEW-2000', 'OLD-1000']);
    }

    public function test_plate_search_filters_results(): void
    {
        Violation::factory()->create(['plate_number' => 'DHAKA-GA-1234']);
        Violation::factory()->create(['plate_number' => 'CTG-CHA-9876']);

        $this->actingAs($this->officer())
            ->get('/violations?search=DHAKA')
            ->assertOk()
            ->assertSee('DHAKA-GA-1234')
            ->assertDontSee('CTG-CHA-9876');
    }

    public function test_type_color_and_date_filters_work_together(): void
    {
        Violation::factory()->create([
            'plate_number' => 'MATCH-1234',
            'violation_type' => 'RED_LIGHT',
            'vehicle_color' => 'RED',
            'frame_timestamp' => '2026-07-03 10:00:00',
        ]);
        Violation::factory()->create([
            'plate_number' => 'WRONG-5678',
            'violation_type' => 'OVERSPEED',
            'vehicle_color' => 'BLACK',
            'frame_timestamp' => '2026-07-03 10:00:00',
        ]);

        $query = http_build_query([
            'type' => 'RED_LIGHT',
            'color' => 'RED',
            'date_from' => '2026-07-03',
            'date_to' => '2026-07-03',
        ]);

        $this->actingAs($this->officer())
            ->get("/violations?{$query}")
            ->assertOk()
            ->assertSee('MATCH-1234')
            ->assertDontSee('WRONG-5678');
    }

    public function test_status_filter_only_shows_matching_reviews(): void
    {
        Violation::factory()->create(['plate_number' => 'PENDING-1000', 'status' => 'PENDING']);
        Violation::factory()->create(['plate_number' => 'CONFIRMED-2000', 'status' => 'CONFIRMED']);

        $this->actingAs($this->officer())
            ->get('/violations?status=CONFIRMED')
            ->assertOk()
            ->assertSee('CONFIRMED-2000')
            ->assertDontSee('PENDING-1000');
    }
}
