<?php

namespace App\Http\Controllers;

use App\Models\Violation;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Validation\Rule;
use Illuminate\View\View;

class ViolationController extends Controller
{
    /**
     * Display a searchable, filterable violation evidence register.
     */
    public function index(Request $request): View
    {
        $filters = $request->validate([
            'search' => ['nullable', 'string', 'max:100'],
            'type' => ['nullable', Rule::in(['OVERSPEED', 'RED_LIGHT'])],
            'color' => ['nullable', 'string', 'max:24'],
            'date_from' => ['nullable', 'date'],
            'date_to' => ['nullable', 'date', 'after_or_equal:date_from'],
            'status' => ['nullable', Rule::in(['PENDING', 'CONFIRMED', 'DISMISSED'])],
        ]);

        $violations = Violation::query()
            ->when($filters['search'] ?? null, function ($query, string $search) {
                $query->where(function ($nested) use ($search) {
                    $nested->where('plate_number', 'like', "%{$search}%")
                        ->orWhere('event_id', 'like', "%{$search}%");
                });
            })
            ->when($filters['type'] ?? null,
                fn ($query, string $type) => $query->where('violation_type', $type))
            ->when($filters['color'] ?? null,
                fn ($query, string $color) => $query->where('vehicle_color', $color))
            ->when($filters['date_from'] ?? null,
                fn ($query, string $date) => $query->whereDate('frame_timestamp', '>=', $date))
            ->when($filters['date_to'] ?? null,
                fn ($query, string $date) => $query->whereDate('frame_timestamp', '<=', $date))
            ->when($filters['status'] ?? null,
                fn ($query, string $status) => $query->where('status', $status))
            ->latest('frame_timestamp')
            ->paginate(15)
            ->withQueryString();

        $summary = [
            'total' => Violation::count(),
            'overspeed' => Violation::where('violation_type', 'OVERSPEED')->count(),
            'red_light' => Violation::where('violation_type', 'RED_LIGHT')->count(),
            'pending' => Violation::where('status', 'PENDING')->count(),
        ];

        $colors = Violation::query()
            ->whereNotNull('vehicle_color')
            ->where('vehicle_color', '!=', 'UNKNOWN')
            ->distinct()
            ->orderBy('vehicle_color')
            ->pluck('vehicle_color');

        return view('violations.index', compact('violations', 'summary', 'colors'));
    }

    /**
     * Display one complete evidence record.
     */
    public function show(Violation $violation): View
    {
        $violation->load('reviewer');

        return view('violations.show', compact('violation'));
    }

    /**
     * Save an officer's plate correction, notes, and review decision.
     */
    public function update(Request $request, Violation $violation): RedirectResponse
    {
        $validated = $request->validate([
            'plate_number' => ['required', 'string', 'max:100'],
            'status' => ['required', Rule::in(['PENDING', 'CONFIRMED', 'DISMISSED'])],
            'officer_notes' => ['nullable', 'string', 'max:2000'],
        ]);

        $plateNumber = preg_replace('/\s+/u', ' ', trim($validated['plate_number']));

        DB::transaction(function () use ($request, $violation, $validated, $plateNumber) {
            if ($plateNumber !== $violation->plate_number) {
                $violation->original_plate_number ??= $violation->plate_number;
                $violation->plate_number = $plateNumber;
                $violation->plate_corrected_at = now();
            }

            $violation->status = $validated['status'];
            $violation->officer_notes = $validated['officer_notes'] ?? null;
            $violation->reviewed_by = $request->user()->id;
            $violation->reviewed_at = $validated['status'] === 'PENDING' ? null : now();
            $violation->save();
        });

        return redirect()
            ->route('violations.show', $violation)
            ->with('status', 'Evidence review saved.');
    }
}
